//===-- ApplyPatterns.cpp ---------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/ApplyPatterns.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDL/IR/PDLOps.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h"  // IWYU pragma: keep

#define DEBUG_TYPE "ktdfarch-apply-patterns"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace mlir::ktdf_arch {
#define GEN_PASS_DEF_APPLYPATTERNSPASS
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf_arch

//===----------------------------------------------------------------------===//
// Native Constraints and Rewrites
//===----------------------------------------------------------------------===//

namespace {

auto ktdfArchMappedTo(PatternRewriter& /*rewriter*/, PDLResultList& results,
                      ArrayRef<PDLValue> values) -> LogicalResult {
  assert(values.size() == 1);

  auto* const op = values[0].cast<Operation*>();
  const auto maps_to = getProperty<MapsToAttr>(op);
  if (!maps_to) {
    return failure();
  }

  results.push_back(maps_to);
  return success();
}

auto ktdfArchHasFeature(PatternRewriter& /*rewriter*/, PDLResultList& results,
                        ArrayRef<PDLValue> values) -> LogicalResult {
  assert(values.size() >= 2 && values.size() <= 3);

  auto* const op = values[0].cast<Operation*>();
  const auto name = cast<StringAttr>(values[1].cast<Attribute>());

  std::optional<Feature> provided;
  if (values.size() == 2) {
    provided = getFeature(op, name);
  } else {
    provided = getFeature(op, {name, values[2].cast<Attribute>()});
  }

  if (!provided) {
    return failure();
  }

  results.push_back(provided->getValue());
  return success();
}

void ktdfArchMapTo(PatternRewriter& rewriter, Operation* op,
                   Attribute maps_to) {
  rewriter.modifyOpInPlace(
      op, [&]() { setProperty(op, cast<MapsToAttr>(maps_to)); });
}

void ktdfArchSetFeature(PatternRewriter& rewriter, Operation* op,
                        Attribute name, Attribute value) {
  rewriter.modifyOpInPlace(op, [&]() {
    NamedAttrList features(
        static_cast<DictionaryAttr>(getProperty<FeaturesAttr>(op)));
    features.set(cast<StringAttr>(name), value);
    setProperty(op,
                cast<FeaturesAttr>(features.getDictionary(op->getContext())));
  });
}

}  // namespace

void ktdf_arch::registerNativeFunctions(PDLPatternModule& patterns) {
  // Constraint functions
  patterns.registerConstraintFunction("ktdf_arch.mapped_to", ktdfArchMappedTo);
  patterns.registerConstraintFunction("ktdf_arch.has_feature",
                                      ktdfArchHasFeature);

  // Rewrite functions
  patterns.registerRewriteFunction("ktdf_arch.map_to", ktdfArchMapTo);
  patterns.registerRewriteFunction("ktdf_arch.set_feature", ktdfArchSetFeature);
}

//===----------------------------------------------------------------------===//
// ktdf_arch::getPatterns
//===----------------------------------------------------------------------===//

namespace {

auto clonePatterns(PatternsOp from, PDLPatternModule& to) -> size_t {
  size_t result = 0U;

  OpBuilder builder(to.getModule().getBodyRegion());
  for (auto pattern : from.getOps<pdl::PatternOp>()) {
    builder.clone(*pattern);
    ++result;
  }

  return result;
}

}  // namespace

auto ktdf_arch::getPatterns(const Device& device, PDLPatternModule& patterns,
                            const StringSet<>& enabled_groups) -> size_t {
  // Ensure there is a module to clone into.
  if (!patterns.getModule()) {
    patterns.mergeIn(PDLPatternModule(ModuleOp::create(device.getLoc())));
  }

  // Clone the selected patterns into the module. Since PDLPatternModule needs
  // ownership of the pattern ops, there is no better way of caching these.
  size_t result = 0U;
  for (auto pattern_set : device.getBodyRegion().getOps<PatternsOp>()) {
    const auto groups = pattern_set.getGroups().getAsValueRange<StringAttr>();
    if (!groups.empty() && llvm::none_of(groups, [&](StringRef group) -> bool {
          return enabled_groups.contains(group);
        })) {
      continue;
    }

    result += clonePatterns(pattern_set, patterns);
  }

  registerNativeFunctions(patterns);
  return result;
}

auto ktdf_arch::getPatterns(const Device& device, PDLPatternModule& patterns,
                            ArrayRef<StringRef> enabled_groups) -> size_t {
  return getPatterns(device, patterns,
                     StringSet<>(llvm::from_range, enabled_groups));
}

//===----------------------------------------------------------------------===//
// ApplyPatternsPass
//===----------------------------------------------------------------------===//

namespace {

struct ApplyPatternsPass
    : public ktdf_arch::impl::ApplyPatternsPassBase<ApplyPatternsPass> {
  using ApplyPatternsPassBase::ApplyPatternsPassBase;

  void runOnOperation() override {
    // Find the device that this function maps to.
    auto declaration = findDeviceDeclarationFor(getOperation());
    if (!declaration) {
      getOperation()->emitError("unable to locate device");
      signalPassFailure();
      return;
    }
    DeviceRef device(declaration, getAnalysisManager());

    LDBG() << "processing "
           << OpWithFlags(getOperation(), OpPrintingFlags().skipRegions())
           << " with device " << declaration.getName();

    // Collect all the patterns to be applied.
    PDLPatternModule pdl_patterns;
    const auto num_patterns = getPatterns(
        device, pdl_patterns, StringSet<>(llvm::from_range, enabled_groups));
    if (num_patterns == 0) {
      return;
    }
    LDBG_OS([&](llvm::raw_ostream& os) {
      os << "applying " << num_patterns << " pattern(s)";
      if (!enabled_groups.empty()) {
        os << " in group(s) ";
        llvm::interleaveComma(enabled_groups, os);
      }
    });

    // Run all the patterns.
    auto changed = false;
    if (failed(applyPatternsGreedily(
            getOperation(), FrozenRewritePatternSet(std::move(pdl_patterns)),
            GreedyRewriteConfig(), &changed))) {
      signalPassFailure();
      return;
    }

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

}  // namespace
