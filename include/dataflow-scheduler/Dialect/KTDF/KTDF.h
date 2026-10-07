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

using Token = TypedValue<TokenType>;
using FifoSlot = TypedValue<FifoSlotType>;

/// RAII helper that allows moving code to a PrivateOp.
///
/// Users may call `tryMakePrivate` on ops to attempt making them private. The
/// updating of the PrivateOp is deffered until the `build()` method is called
/// or the helper is destroyed. The resulting PrivateOp is in canonical form.
///
/// Alternatively, private operations may be inserted by using this instance
/// as the builder argument to `create`. Changing the insertion block, however,
/// leads to undefined behavior.
class PrivateBuilder : public ImplicitLocOpBuilder {
 public:
  /// Canonicalizes @p op .
  ///
  /// - Results without users are dropped.
  /// - Values yielded multiple times are coalesced into one result.
  /// - External yielded values replace their results.
  /// - If the resulting PrivateOp is empty, it is erased.
  static void canonicalize(RewriterBase& rewriter, PrivateOp op) {
    op = PrivateBuilder(op, rewriter.getListener()).build();
    if (op.getBody()->without_terminator().empty()) {
      rewriter.eraseOp(op);
    }
  }
  /// Canonicalizes the PrivateOp of @p pipeline , if any.
  ///
  /// See canonicalize(RewriterBase&, PrivateOp) for more details.
  static void canonicalize(RewriterBase& rewriter, PipelineOp pipeline) {
    if (auto existing = pipeline.getPrivateOp(); existing) {
      canonicalize(rewriter, existing);
    }
  }

  /// Initializes a builder for @p existing .
  explicit PrivateBuilder(PrivateOp existing,
                          OpBuilder::Listener* listener = nullptr);
  /// Initializes a builder for @p pipeline , inserting a PrivateOp if needed.
  explicit PrivateBuilder(PipelineOp pipeline,
                          std::optional<Location> loc = std::nullopt,
                          OpBuilder::Listener* listener = nullptr);
  /// Initializes a builder for @p pipeline , inserting a PrivateOp if needed.
  explicit PrivateBuilder(const OpBuilder& builder, PipelineOp pipeline,
                          std::optional<Location> loc = std::nullopt)
      : PrivateBuilder(pipeline, loc, builder.getListener()) {}
  /// Initializes a builder for @p pipeline , inserting a PrivateOp if needed.
  explicit PrivateBuilder(const ImplicitLocOpBuilder& builder,
                          PipelineOp pipeline)
      : PrivateBuilder(pipeline, builder.getLoc(), builder.getListener()) {}

  /// Ensures the PrivateOp is built.
  virtual ~PrivateBuilder() { build(); }

  PrivateBuilder(PrivateBuilder&&) = delete;
  PrivateBuilder(const PrivateBuilder&) = delete;
  auto operator=(PrivateBuilder&&) = delete;
  auto operator=(const PrivateBuilder&) = delete;

  // Changing the insertion block leads to undefined behavior.
  void setInsertionPoint() = delete;
  void setInsertionPointAfter() = delete;
  void setInsertionPointToStart() = delete;
  void setInsertionPointToEnd() = delete;
  void setInsertionPointAfterValue() = delete;
  void clearInsertionPoint() = delete;
  void restoreInsertionPoint() = delete;

  /// Determines whether @p block is in the PrivateOp.
  [[nodiscard]] auto isPrivate(Block* block) const -> bool {
    return block == getInsertionBlock();
  }
  /// Determines whether @p op is in the PrivateOp.
  [[nodiscard]] auto isPrivate(Operation* op) const -> bool {
    return isPrivate(op->getBlock());
  }

  /// Determines whether @p op can be moved into the PrivateOp.
  ///
  /// Checks whether it is legal to hoist @p op out of its current scope and
  /// then sink it into the PrivateOp.
  [[nodiscard]] virtual auto canPrivate(Operation* op) const -> bool;

  /// Moves @p op into the PrivateOp.
  ///
  /// @pre  `canPrivate(op)`
  virtual void makePrivate(Operation* op);

  /// Attempts to move @p op into the PrivateOp.
  ///
  /// Privating fails if the SSA property would be broken by moving @p op :
  /// - @p op may not have users that can't access the PrivateOp results.
  /// - @p op may not hold uses that aren't available in the PrivateOp.
  ///
  /// @return Whether @p op was privated.
  auto tryMakePrivate(Operation* op) -> LogicalResult {
    if (isPrivate(op)) {
      return success();
    }
    if (canPrivate(op)) {
      makePrivate(op);
      return success();
    }
    return failure();
  }

  /// Creates a new private token.
  [[nodiscard]] auto createToken(std::optional<Location> loc = std::nullopt)
      -> Token;
  /// Creates a new private FIFO.
  [[nodiscard]] auto createFifo(ArrayRef<FifoSlotType> slots,
                                ValueRange dynamic_sizes = {},
                                std::optional<Location> loc = std::nullopt)
      -> ValueRange;

  /// Builds the PrivateOp.
  ///
  /// Applies all deferred modifications to the IR and finalizes the result.
  /// The PrivateBuilder is left in a state as if it was re-initialized on the
  /// resulting operation. Returns `nullptr` once the PrivateOp was erased.
  auto build() -> PrivateOp;

  /// Erases the PrivateOp. Nothing can be inserted afterwards.
  void erase();
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDF_H_
