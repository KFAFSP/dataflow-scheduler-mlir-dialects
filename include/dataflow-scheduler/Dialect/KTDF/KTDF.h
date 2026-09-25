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

#include <llvm/Support/PointerLikeTypeTraits.h>
#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"  // IWYU pragma: keep

namespace mlir::ktdf {

/// Resource used for ktdf FIFO queues.
struct FifoResource : public mlir::SideEffects::Resource::Base<FifoResource> {
  [[nodiscard]] auto getName() -> StringRef final { return "KTDFFIFOResource"; }
};

}  // namespace mlir::ktdf

/// Auto-generated includes.
#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h.inc"

template <>
struct llvm::PointerLikeTypeTraits<mlir::ktdf::StageOp>
    : PointerLikeTypeTraits<mlir::Operation*> {
  [[nodiscard]] static auto getFromVoidPointer(void* ptr)
      -> mlir::ktdf::StageOp {
    return mlir::ktdf::StageOp::getFromOpaquePointer(ptr);
  }
};

namespace mlir::ktdf {

/// RAII helper that allows moving code to a PrivateOp in a PipelineOp.
///
/// Users may call `makePrivate` on operations to attempt making them private
/// to the pipeline. The move is deferred until the helper is destroyed. The
/// helper allows for both hoisting and sinking of ops into the PrivateOp.
class PipelinePrivatizer : protected RewriterBase {
 public:
  using Token = TypedValue<TokenType>;
  using FifoSlot = TypedValue<FifoSlotType>;

  /// Canonicalizes the PrivateOp of @p op .
  ///
  /// - Results without users are dropped.
  /// - Values yielded multiple times are coalesced into one result.
  /// - External yielded values replace their results.
  /// - If the resulting PrivateOp is empty, it is erased.
  static void canonicalize(PipelineOp op,
                           OpBuilder::Listener* listener = nullptr) {
    PipelinePrivatizer(op, true, listener);
  }
  /// @copydoc canonicalize(PipelineOp, OpBuilder::Listener*)
  static void canonicalize(const RewriterBase& rewriter, PipelineOp op) {
    PipelinePrivatizer(rewriter, op, true);
  }

  /// Creates a PipelinePrivatizer for @p pipeline .
  ///
  /// If @p force_recreate is `true`, any existing PrivateOp will be re-created
  /// in its canonical form, even if no modifications are made.
  explicit PipelinePrivatizer(PipelineOp pipeline, bool force_recreate = false,
                              OpBuilder::Listener* listener = nullptr);
  /// @copydoc PipelinePrivatizer(PipelineOp, bool, OpBuilder::Listener*)
  explicit PipelinePrivatizer(const OpBuilder& builder, PipelineOp pipeline,
                              bool force_recreate = false)
      : PipelinePrivatizer(pipeline, force_recreate, builder.getListener()) {}

  /// Finalizes the outstanding modifications to the pipeline.
  ~PipelinePrivatizer() override { finalize(); }

  PipelinePrivatizer(PipelinePrivatizer&&) = delete;
  PipelinePrivatizer(const PipelinePrivatizer&) = delete;

  auto operator=(PipelinePrivatizer&&) = delete;
  auto operator=(const PipelinePrivatizer&) = delete;

  /// Determines whether @p block will be within the PrivateOp.
  [[nodiscard]] auto isPrivate(Block* block) const -> bool;
  /// Determines whether @p op will be within the PrivateOp.
  [[nodiscard]] auto isPrivate(Operation* op) const -> bool {
    return isPrivate(op->getBlock());
  }

  /// Gets the MLIRContext.
  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return OpBuilder::getContext();
  }
  /// Gets the underlying pipeline.
  [[nodiscard]] auto getPipeline() const -> PipelineOp { return pipeline_; }

  /// Attempts to make @p op a private result.
  ///
  /// Privating fails if the SSA property would be broken by moving @p op :
  /// - If @p op is defined inside the pipeline, it may not be nested within an
  ///   isolated region, and its operand definitions must reach the pipeline.
  /// - If @p op is defined outside the pipeline, it must not have any users
  ///   outside the pipeline.
  ///
  /// @return Whether @p op was privated.
  auto makePrivate(Operation* op) -> LogicalResult;

  /// Creates a new token inside the private region.
  [[nodiscard]] auto createToken(std::optional<Location> loc = std::nullopt)
      -> Token;
  /// Creates a new FIFO inside the private region.
  [[nodiscard]] auto createFifo(ArrayRef<FifoSlotType> slots,
                                ValueRange dynamic_sizes = {},
                                std::optional<Location> loc = std::nullopt)
      -> ValueRange;

  /// Finalizes the outstanding modifications to the pipeline.
  ///
  /// If there are no modifications to perform, does nothing. After finalizing,
  /// the PipelinePrivatizer will be ready again to queue more modifications to
  /// the same pipeline.
  virtual auto finalize() -> PipelineOp;

 private:
  PipelineOp pipeline_;
  PrivateOp existing_;

  /// Unlinked accumulator for operations to be privated on finalize().
  Block private_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDF_H_
