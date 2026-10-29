//===-- TestDataMovementCoalescing.cpp --------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
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
//
// This file implements a test pass for the data movement coalescing patterns.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/DataTransfers.h"

using namespace mlir;

namespace {

struct TestDataMovementCoalescingPass
    : PassWrapper<TestDataMovementCoalescingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataMovementCoalescingPass)

  [[nodiscard]] auto getArgument() const -> llvm::StringRef final {
    return "test-ktdf-coalesce-data-movement";
  }
  [[nodiscard]] auto getDescription() const -> llvm::StringRef final {
    return "Greedily applies the KTDF data movement coalescing patterns";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ktdf::KTDFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ktdf::populateDataMovementCoalescingPatterns(patterns);

    auto changed = false;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
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

namespace mlir::ktdf::test {

void registerTestDataMovementCoalescingPass() {
  PassRegistration<TestDataMovementCoalescingPass>();
}

}  // namespace mlir::ktdf::test
