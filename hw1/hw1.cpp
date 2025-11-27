// LLVM 17 new PM plugin: hw1
// Analyze single-loop array dependences using affine (linear) subscripts via
// solving linear Diophantine equalities within loop bounds, and emit JSON:
// {
//   "FlowDependence": [...],
//   "AntiDependence": [...],
//   "OutputDependence": [...]
// }
// Each entry: {"array":"A","src_stmt":1,"src_idx":4,"dst_stmt":2,"dst_idx":8}

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/ArrayRef.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct Affine {
  // expr = a * i + b
  int64_t a = 0;
  int64_t b = 0;
};

struct Access {
  std::string Array;   // e.g., "A"
  Affine Expr;         // subscript: a*i + b
  bool IsWrite = false;
  int StmtId = 0;      // 1-based statement id in loop body order
};

struct Statement {
  int Id = 0;
  SmallVector<Access, 8> Reads;
  SmallVector<Access, 4> Writes; // usually 1
};

struct DependenceRecord {
  std::string Array;
  int SrcStmt;
  int64_t SrcIdx;
  int DstStmt;
  int64_t DstIdx;
};

// Try to get AllocaInst behind a Value (through bitcasts/gep 0?)
static AllocaInst *getAlloca(Value *V) {
  Value *P = V;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(P))
    P = GEP->getPointerOperand();
  if (auto *BC = dyn_cast<BitCastInst>(P))
    P = BC->getOperand(0);
  return dyn_cast<AllocaInst>(P);
}

// Parse affine form a*i + b with respect to the loop index load(s).
// We treat any LoadInst whose pointer operand equals IAlloca as variable i (coef 1).
static std::optional<Affine> parseAffine(Value *V, AllocaInst *IAlloca) {
  // strip casts/extends
  if (auto *SE = dyn_cast<SExtInst>(V))
    return parseAffine(SE->getOperand(0), IAlloca);
  if (auto *ZE = dyn_cast<ZExtInst>(V))
    return parseAffine(ZE->getOperand(0), IAlloca);

  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    return Affine{0, (int64_t)CI->getSExtValue()};
  }
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    if (LI->getPointerOperand() == IAlloca)
      return Affine{1, 0};
    // Loading from arrays or other pointers -> not affine in i (for index)
    return std::nullopt;
  }
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    auto L = parseAffine(BO->getOperand(0), IAlloca);
    auto R = parseAffine(BO->getOperand(1), IAlloca);
    switch (BO->getOpcode()) {
    case Instruction::Add:
      if (L && R)
        return Affine{L->a + R->a, L->b + R->b};
      break;
    case Instruction::Sub:
      if (L && R)
        return Affine{L->a - R->a, L->b - R->b};
      break;
    case Instruction::Mul: {
      // allow const * affine or affine * const
      if (isa<ConstantInt>(BO->getOperand(0)) && R) {
        auto K = cast<ConstantInt>(BO->getOperand(0))->getSExtValue();
        return Affine{R->a * K, R->b * K};
      }
      if (isa<ConstantInt>(BO->getOperand(1)) && L) {
        auto K = cast<ConstantInt>(BO->getOperand(1))->getSExtValue();
        return Affine{L->a * K, L->b * K};
      }
      break;
    }
    default:
      break; // unsupported op
    }
    return std::nullopt;
  }

  return std::nullopt;
}

// If Ptr is a GEP of the form gep [N x T], %Array, 0, idx, parse idx as affine
static std::optional<std::pair<std::string, Affine>>
parseArraySubscript(Value *Ptr, AllocaInst *IAlloca) {
  auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return std::nullopt;
  if (GEP->getNumIndices() < 2)
    return std::nullopt;

  // base should originate from an alloca (array variable)
  Value *Base = GEP->getPointerOperand();
  if (auto *BC = dyn_cast<BitCastInst>(Base))
    Base = BC->getOperand(0);
  auto *AI = dyn_cast<AllocaInst>(Base);
  if (!AI)
    return std::nullopt;
  // Expect array type [N x T]
  auto *PTy = dyn_cast<PointerType>(AI->getType());
  if (!PTy)
    return std::nullopt;

  // last index is the subscript
  Value *Idx = nullptr;
  // For LLVM 17, GEP has source element type; indices via iterator
  unsigned idxCount = 0;
  for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; ++I) {
    if (++idxCount == 2) { // the second index for [0, i]
      Idx = I->get();
      break;
    }
  }
  if (!Idx)
    return std::nullopt;

  auto Aff = parseAffine(Idx, IAlloca);
  if (!Aff)
    return std::nullopt;

  std::string Name = AI->getName().str();
  // strip possible leading % or digits not likely, but names are clean (A,B,C,D)
  return std::make_pair(Name, *Aff);
}

// Derive loop bounds [L, U) from IR with variable %i alloca.
static std::optional<std::pair<int64_t, int64_t>>
getLoopBounds(Function &F, AllocaInst *IAlloca) {
  // Lower bound: a store constant to %i in entry block
  int64_t L = 0;
  bool LFound = false;
  if (auto *Entry = &F.getEntryBlock()) {
    for (Instruction &I : *Entry) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (SI->getPointerOperand() == IAlloca) {
          if (auto *CI = dyn_cast<ConstantInt>(SI->getValueOperand())) {
            L = (int64_t)CI->getSExtValue();
            LFound = true;
            break;
          }
        }
      }
    }
  }
  // Upper bound: icmp slt (load %i), Const
  int64_t U = 0;
  bool UFound = false;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *IC = dyn_cast<ICmpInst>(&I)) {
        if (IC->getPredicate() == CmpInst::ICMP_SLT ||
            IC->getPredicate() == CmpInst::ICMP_ULT) {
          Value *Op0 = IC->getOperand(0);
          Value *Op1 = IC->getOperand(1);
          LoadInst *L0 = dyn_cast<LoadInst>(Op0);
          if (L0 && L0->getPointerOperand() == IAlloca) {
            if (auto *CI = dyn_cast<ConstantInt>(Op1)) {
              U = (int64_t)CI->getSExtValue();
              UFound = true;
            }
          }
        }
      }
    }
  }
  if (!LFound || !UFound)
    return std::nullopt;
  return std::make_pair(L, U);
}

// Find the loop body basic block: try named "for.body", else choose the unique block
// (other than entry/exit) containing stores to arrays.
static BasicBlock *findLoopBody(Function &F) {
  for (BasicBlock &BB : F) {
    if (BB.getName().startswith("for.body"))
      return &BB;
  }
  // Fallback: the block with multiple loads/stores and a branch to an increment block
  BasicBlock *Candidate = nullptr;
  for (BasicBlock &BB : F) {
    unsigned Stores = 0;
    for (Instruction &I : BB) {
      if (isa<StoreInst>(I))
        Stores++;
    }
    if (Stores >= 1) {
      Candidate = &BB;
      break;
    }
  }
  return Candidate;
}

// Collect statements in loop body in source order: we group all loads since the
// previous store as reads of the current statement, and the store as its write.
static SmallVector<Statement,16> collectStatements(BasicBlock *Body, AllocaInst *IAlloca, StringSet<> &IndVarNames) {
  SmallVector<Statement,16> Stmts;
  SmallVector<Access,16> PendingReads;
  int sid = 0;

  auto flushReads = [&]() { PendingReads.clear(); };

  for (Instruction &I : *Body) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (auto Parsed = parseArraySubscript(LI->getPointerOperand(), IAlloca)) {
        Access A;
        A.Array = Parsed->first;
        A.Expr = Parsed->second;
        A.IsWrite = false;
        // stmt id will be filled when attached to a statement
        PendingReads.push_back(A);
      }
      continue;
    }
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      auto Parsed = parseArraySubscript(SI->getPointerOperand(), IAlloca);
      if (!Parsed) {
        // store to scalar or unknown, ignore but still part of body
        continue;
      }
      // New statement
      Statement S;
      S.Id = ++sid;
      // attach reads
      for (auto &R : PendingReads) {
        R.StmtId = S.Id;
        S.Reads.push_back(R);
      }
      flushReads();
      // attach the write
      Access W;
      W.Array = Parsed->first;
      W.Expr = Parsed->second;
      W.IsWrite = true;
      W.StmtId = S.Id;
      S.Writes.push_back(W);

      Stmts.push_back(std::move(S));
    }
  }
  return Stmts;
}

// Solve equality a1*t + b1 = a2*s + b2 with s,t in [L,U) and order constraint cmp(s,t).
// We enumerate within bounds (small loops) which is equivalent to bounded Diophantine solving.
// Extended Euclid: returns gcd plus Bezout coefficients: a*x + b*y = gcd
static void extendedEuclid(int64_t a, int64_t b, int64_t &g, int64_t &x, int64_t &y) {
  if (b == 0) { g = (a >= 0 ? a : -a); x = (a >= 0 ? 1 : -1); y = 0; return; }
  int64_t x1=0,y1=0; extendedEuclid(b, a % b, g, x1, y1);
  x = y1;
  y = x1 - (a / b) * y1;
}

// Solve a*T + b*S = c (Diophantine) produce bounded integer solutions (T,S) with L <= T,S < U
// Using general solution: T = T0 + (b/g)*k, S = S0 - (a/g)*k
// cmp(S,T) must hold for dependence ordering.
template <typename Cmp>
static SmallVector<std::pair<int64_t,int64_t>,32> solveBoundedEqual(const Affine &W, const Affine &R,
                                                                    int64_t L, int64_t U, Cmp cmp) {
  // Equation: W.a * t - R.a * s = R.b - W.b  -> a*t + b*s = c with a = W.a, b = -R.a
  int64_t a = W.a;
  int64_t b = -R.a;
  int64_t c = R.b - W.b;
  SmallVector<std::pair<int64_t,int64_t>,32> out;

  // Handle trivial cases
  if (a == 0 && b == 0) {
    if (c == 0) {
      // Any (t,s) with equality holds: enumerate constrained & ordering
      for (int64_t t=L; t<U; ++t) for (int64_t s=L; s<U; ++s) if (cmp(s,t)) out.emplace_back(t,s);
      std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    return out;
  }
  if (a == 0) { // b*s = c -> s = c/b, t free but must satisfy ordering and bounds
    if (c % b != 0) return out;
    int64_t s = c / b;
    if (s < L || s >= U) return out;
    for (int64_t t=L; t<U; ++t) if (cmp(s,t)) out.emplace_back(t,s);
    std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }
  if (b == 0) { // a*t = c -> t = c/a, s free
    if (c % a != 0) return out;
    int64_t t = c / a;
    if (t < L || t >= U) return out;
    for (int64_t s=L; s<U; ++s) if (cmp(s,t)) out.emplace_back(t,s);
    std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

  int64_t g=0,x0=0,y0=0; extendedEuclid(a,b,g,x0,y0); // a*x0 + b*y0 = g
  if (c % g != 0) return out; // no solution
  int64_t k = c / g;
  // particular solution
  int64_t t0 = x0 * k;
  int64_t s0 = y0 * k;
  // general solution coefficients
  int64_t tCoeff = b / g;
  int64_t sCoeff = -a / g;

  // Determine k range from bounds:
  // L <= t0 + tCoeff * q < U  and  L <= s0 + sCoeff * q < U
  auto computeRange = [&](int64_t base, int64_t coeff) -> std::pair<int64_t,int64_t> {
    if (coeff == 0) {
      if (base < L || base >= U) return {1,0}; // empty range
      return {INT64_MIN/4, INT64_MAX/4}; // any q acceptable
    }
    // Solve L <= base + coeff*q < U
    // Lower bound: base + coeff*q >= L -> coeff*q >= L - base
    // Upper bound: base + coeff*q < U -> coeff*q < U - base
    double q1 = (double)(L - base) / (double)coeff;
    double q2 = (double)(U - 1 - base) / (double)coeff; // inclusive upper: base+coeff*q <= U-1
    int64_t lo, hi;
    if (coeff > 0) {
      lo = (int64_t)ceil(q1); hi = (int64_t)floor(q2);
    } else { // coeff < 0
      lo = (int64_t)ceil(q2); hi = (int64_t)floor(q1);
    }
    return {lo, hi};
  };

  auto [q1lo,q1hi] = computeRange(t0, tCoeff);
  auto [q2lo,q2hi] = computeRange(s0, sCoeff);
  int64_t lo = std::max(q1lo, q2lo);
  int64_t hi = std::min(q1hi, q2hi);
  if (lo > hi) return out; // empty intersection

  // Iterate q in intersection
  for (int64_t q = lo; q <= hi; ++q) {
    int64_t t = t0 + tCoeff * q;
    int64_t s = s0 + sCoeff * q;
    if (t < L || t >= U || s < L || s >= U) continue; // safety
    if (!cmp(s,t)) continue;
    out.emplace_back(t,s);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Create JSON file name: prefer Module.getSourceFileName() base with .json
static std::string deriveJsonName(const Module &M) {
  std::string base;
  if (!M.getSourceFileName().empty()) {
    base = M.getSourceFileName(); // e.g., "test1.c"
  } else {
    base = M.getModuleIdentifier();
  }
  // strip directory
  auto posSlash = base.find_last_of("/\\");
  if (posSlash != std::string::npos)
    base = base.substr(posSlash + 1);
  // replace extension with .json
  auto dot = base.find_last_of('.');
  if (dot != std::string::npos)
    base = base.substr(0, dot);
  base += ".json";
  return base;
}

class HW1Pass : public PassInfoMixin<HW1Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    // Assume single function of interest
    Function *TargetF = nullptr;
    for (Function &F : M) {
      if (!F.isDeclaration()) {
        TargetF = &F;
        break;
      }
    }
    if (!TargetF)
      return PreservedAnalyses::all();

    // Find loop var alloca named "i" (or the first i32 alloca used as loop counter)
    AllocaInst *IAlloca = nullptr;
    for (Instruction &I : TargetF->getEntryBlock()) {
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        if (AI->getAllocatedType()->isIntegerTy(32)) {
          if (AI->getName() == "i" || !IAlloca)
            IAlloca = AI;
        }
      }
    }
    if (!IAlloca)
      return PreservedAnalyses::all();

    auto Bounds = getLoopBounds(*TargetF, IAlloca);
    if (!Bounds)
      return PreservedAnalyses::all();
    int64_t L = Bounds->first;
    int64_t U = Bounds->second; // exclusive

    BasicBlock *Body = findLoopBody(*TargetF);
    if (!Body)
      return PreservedAnalyses::all();

  StringSet<> IndVarNames;
  IndVarNames.insert(IAlloca->getName());
  auto Stmts = collectStatements(Body, IAlloca, IndVarNames);
    // Debug (disabled): dump collected statements
    // errs() << "[hw1] Collected " << (unsigned)Stmts.size() << " statements\n";
    // for (const auto &S : Stmts) {
    //   errs() << "  Stmt " << S.Id << "\n";
    //   for (const auto &R : S.Reads)
    //     errs() << "    read  " << R.Array << " : (" << R.Expr.a << ") i + (" << R.Expr.b << ")\n";
    //   for (const auto &W : S.Writes)
    //     errs() << "    write " << W.Array << " : (" << W.Expr.a << ") i + (" << W.Expr.b << ")\n";
    // }

    // Index statements by id for easy lookup
    // Build access maps by array name
    // We'll generate dependences:
  SmallVector<DependenceRecord,64> Flow;
  SmallVector<DependenceRecord,64> Anti;
  SmallVector<DependenceRecord,64> Output;

    // Build read/write sets by statement and array
    for (size_t a = 0; a < Stmts.size(); ++a) {
      const auto &Sa = Stmts[a];
      for (size_t b = 0; b < Stmts.size(); ++b) {
        const auto &Sb = Stmts[b];
        // Define order predicate for same-iteration relations
        auto earlierInBody = (Sa.Id < Sb.Id);

        // Flow: Sa.write -> Sb.read (same array)
        for (const auto &Wa : Sa.Writes) {
          for (const auto &Rb : Sb.Reads) {
            if (Wa.Array != Rb.Array)
              continue;
            auto sols = solveBoundedEqual(Wa.Expr, Rb.Expr, L, U,
              [&](int64_t s, int64_t t) {
                // read at iteration s after write at iteration t
                return (s > t) || (s == t && earlierInBody);
              });
            for (auto [t, s] : sols) {
              Flow.push_back({Wa.Array, Sa.Id, t, Sb.Id, s});
            }
          }
        }

        // Anti: Sa.read -> Sb.write (same array)
        for (const auto &Ra : Sa.Reads) {
          for (const auto &Wb : Sb.Writes) {
            if (Ra.Array != Wb.Array)
              continue;
            auto sols = solveBoundedEqual(Wb.Expr, Ra.Expr, L, U,
              [&](int64_t s, int64_t t) {
                // write at iteration t after read at iteration s
                // Here cmp uses (s,t) with Ra at s, Wb at t
                return (t > s) || (t == s && (Sa.Id < Sb.Id));
              });
            for (auto [t, s] : sols) {
              // Note: src is read (Sa, s), dst is write (Sb, t)
              Anti.push_back({Ra.Array, Sa.Id, s, Sb.Id, t});
            }
          }
        }

        // Output: Sa.write -> Sb.write (same array)
        for (const auto &Wa : Sa.Writes) {
          for (const auto &Wb : Sb.Writes) {
            if (Wa.Array != Wb.Array)
              continue;
            auto sols = solveBoundedEqual(Wa.Expr, Wb.Expr, L, U,
              [&](int64_t s, int64_t t) {
                // second write (Sb at s) occurs after first (Sa at t)
                return (s > t) || (s == t && earlierInBody);
              });
            for (auto [t, s] : sols) {
              Output.push_back({Wa.Array, Sa.Id, t, Sb.Id, s});
            }
          }
        }
      }
    }

    // Sort and unique to avoid duplicates
    auto depLess = [](const DependenceRecord &x, const DependenceRecord &y) {
      if (x.Array != y.Array) return x.Array < y.Array;
      if (x.SrcStmt != y.SrcStmt) return x.SrcStmt < y.SrcStmt;
      if (x.SrcIdx != y.SrcIdx) return x.SrcIdx < y.SrcIdx;
      if (x.DstStmt != y.DstStmt) return x.DstStmt < y.DstStmt;
      return x.DstIdx < y.DstIdx;
    };
    auto uniq = [&](SmallVector<DependenceRecord,64> &V) {
      std::sort(V.begin(), V.end(), depLess);
      V.erase(std::unique(V.begin(), V.end(), [](const DependenceRecord &a, const DependenceRecord &b) {
        return a.Array == b.Array && a.SrcStmt == b.SrcStmt && a.SrcIdx == b.SrcIdx &&
               b.DstStmt == a.DstStmt && b.DstIdx == a.DstIdx;
      }), V.end());
    };
    uniq(Flow); uniq(Anti); uniq(Output);

    // Emit JSON file in current directory, named from source file
    std::string jsonName = deriveJsonName(M);
    std::error_code EC;
    std::string path = jsonName; // cwd
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
      errs() << "[hw1] Failed to open output file: " << path << "\n";
      return PreservedAnalyses::all();
    }

    auto emitArray = [&](ArrayRef<DependenceRecord> V) {
      ofs << "[";
      if (!V.empty()) ofs << "\n";
      for (size_t i = 0; i < V.size(); ++i) {
        const auto &d = V[i];
        ofs << "    {\n";
        ofs << "      \"array\": \"" << d.Array << "\",\n";
        ofs << "      \"src_stmt\": " << d.SrcStmt << ",\n";
        ofs << "      \"src_idx\": " << d.SrcIdx << ",\n";
        ofs << "      \"dst_stmt\": " << d.DstStmt << ",\n";
        ofs << "      \"dst_idx\": " << d.DstIdx << "\n";
        ofs << "    }";
        if (i + 1 != V.size()) ofs << ",\n"; else ofs << "\n";
      }
      ofs << "]";
    };

    ofs << "{\n";
    ofs << "  \"FlowDependence\": ";
  emitArray(ArrayRef<DependenceRecord>(Flow));
    ofs << ",\n";
    ofs << "  \"AntiDependence\": ";
  emitArray(ArrayRef<DependenceRecord>(Anti));
    ofs << ",\n";
    ofs << "  \"OutputDependence\": ";
  emitArray(ArrayRef<DependenceRecord>(Output));
    ofs << "\n}\n";
    ofs.close();

    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hw1", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hw1") {
                    MPM.addPass(HW1Pass());
                    return true;
                  }
                  return false;
                });
          }};
}

 
