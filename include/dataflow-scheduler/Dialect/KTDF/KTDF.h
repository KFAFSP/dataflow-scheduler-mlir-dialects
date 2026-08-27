//===-- KTDF.h --------------------------------------------------*- c++ -*-===//
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
// This file includes the entire ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDF_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDF_H_

#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"  // IWYU pragma: keep

/// Auto-generated includes.
#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h.inc"

namespace mlir::ktdf {

/// RAII helper that allows moving code to a PrivateOp in a PipelineOp.
///
/// Users may call `makePrivate` on operations to attempt making them private
/// to the pipeline. The move is deferred until the helper is destroyed. The
/// helper allows for both hoisting and sinking of ops into the PrivateOp.
class PipelinePrivatizer {
 public:
  /// Canonicalizes the PrivateOp of @p op .
  static void canonicalize(RewriterBase& rewriter, PipelineOp op) {
    PipelinePrivatizer(rewriter, op, true);
  }

  /// Creates a PipelinePrivatizer for @p pipeline .
  ///
  /// If @p force_recreate is `true`, any existing PrivateOp will be re-created
  /// in its canonical form, even if no modifications are made.
  explicit PipelinePrivatizer(RewriterBase& rewriter, PipelineOp pipeline,
                              bool force_recreate = false);
  ~PipelinePrivatizer();

  PipelinePrivatizer(PipelinePrivatizer&&) = delete;
  PipelinePrivatizer(const PipelinePrivatizer&) = delete;

  auto operator=(PipelinePrivatizer&&) = delete;
  auto operator=(const PipelinePrivatizer&) = delete;

  /// Determines whether @p blockwill be within the PrivateOp.
  [[nodiscard]] auto isPrivate(Block* block) -> bool;
  /// Determines whether @p op will be within the PrivateOp.
  [[nodiscard]] auto isPrivate(Operation* op) -> bool {
    return isPrivate(op->getBlock());
  }

  /// Attempts to make @p op a private result.
  ///
  /// Privating fails if the SSA property would be broken by moving @p op :
  /// - If @p op is defined inside the pipeline, it may not be nested within an
  ///   isolated region, and its operand definitions must reach the pipeline.
  /// - If @p op is defined outisde the pipeline, it must not have any users
  ///   outside the pipeline.
  ///
  /// @return Whether @p op was privated.
  auto makePrivate(Operation* op) -> LogicalResult;

 private:
  RewriterBase& rewriter_;
  PipelineOp pipeline_;
  PrivateOp existing_;
  Block private_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDF_H_
