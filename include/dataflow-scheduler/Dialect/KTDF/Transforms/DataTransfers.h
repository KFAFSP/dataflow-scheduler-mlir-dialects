//===-- DataTransfers.h -----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_DATATRANSFERS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_DATATRANSFERS_H_

#include <mlir/IR/AffineMap.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir {
class RewritePatternSet;
}  // namespace mlir

namespace mlir::ktdf {

/// Gets the only producer of @p slot .
///
/// If there is exactly one `ktdf.write_to_fifo` or `ktdf.data_transfer`
/// targeting @p slot (or its private uses), returns that target operand.
///
/// @retval nullptr     Unexpected user or multiple writes to @p slot .
/// @retval OpOperand*  Single producer of @p slot .
[[nodiscard]] auto getProducer(FifoSlot slot) -> OpOperand*;
/// Gets the only producer for @p slot if it dominates @p user in its stage.
///
/// See getProducer(FifoSlot) for more information.
[[nodiscard]] auto getProducer(FifoSlot slot, Operation* user) -> OpOperand*;

/// Gets the only consumer of @p slot .
///
/// If there is exactly one `ktdf.read_from_fifo` or `ktdf.data_transfer`
/// sourcing @p slot (or its private uses), returns that source operand.
///
/// @retval nullptr     Unexpected user or multiple reads from @p slot .
/// @retval OpOperand*  Single consumer of @p slot .
[[nodiscard]] auto getConsumer(FifoSlot slot) -> OpOperand*;

/// References a FifoAllocateOp result.
struct FifoSlotAllocation : OpResult {
  using OpResult::OpResult;

  /// Gets the allocation for @p value , if any.
  [[nodiscard]] static auto get(Value value) -> FifoSlotAllocation {
    auto slot = dyn_cast<FifoSlot>(value);
    return slot ? get(slot) : nullptr;
  }
  /// Gets the allocation for @p value , if any.
  [[nodiscard]] static auto get(FifoSlot slot) -> FifoSlotAllocation {
    return dyn_cast_if_present<FifoSlotAllocation>(getPrivateDefinition(slot));
  }

  [[nodiscard]] static auto classof(Value value) -> bool {
    const auto result = dyn_cast<OpResult>(value);
    return result && classof(result);
  }
  [[nodiscard]] static auto classof(OpResult result) -> bool {
    return isa<FifoAllocateOp>(result.getOwner());
  }

  /*implicit*/ operator FifoSlot() const { return cast<FifoSlot>(*this); }

  /// Gets the FifoAllocateOp that owns this slot.
  [[nodiscard]] auto getOwner() const -> FifoAllocateOp {
    return cast<FifoAllocateOp>(OpResult::getOwner());
  }
  /// Gets the single producer of this slot.
  ///
  /// See mlir::ktdf::getProducer(FifoSlot) for more information.
  [[nodiscard]] auto getProducer() const -> OpOperand* {
    return mlir::ktdf::getProducer(*this);
  }
  /// Gets the single consumer of this slot.
  ///
  /// See mlir::ktdf::getConsumer(FifoSlot) for more information.
  [[nodiscard]] auto getConsumer() const -> OpOperand* {
    return mlir::ktdf::getConsumer(*this);
  }

  /// Gets the FifoSlotType.
  [[nodiscard]] auto getType() const -> FifoSlotType {
    return cast<FifoSlotType>(Value::getType());
  }

  /// Updates the FifoSlotType in place.
  void setType(FifoSlotType type) { Value::setType(type); }

  /// Gets the size of the slot.
  [[nodiscard]] auto getMixedSize() const -> OpFoldResult {
    return getOwner().getMixedSize(getResultNumber());
  }

  /// Resizes the slot in-place, updating any PrivateOp if needed.
  void resize(RewriterBase& rewriter, OpFoldResult num_elements);
};

/// Describes the memory side of a `ktdf.data_transfer`.
struct TransferAccess {
  /// Creates a TransferAccess for @p slice .
  ///
  /// A transfer must always access contiguous memory, so a dim of @p slice may
  /// ony have a non-unit stride if its size is statically known to be 1.
  ///
  /// @retval failure         Non-contiguous @p slice .
  /// @retval TransferAccess  Memory side of a `ktdf.data_transfer`.
  [[nodiscard]] static auto get(MLIRContext* context, ArrayRef<Range> slice)
      -> FailureOr<TransferAccess>;

  AffineMap base_map;
  SmallVector<Value> indices;
  SmallVector<OpFoldResult> sizes;
};

/// Reshapes @p value to @p shape , which must have the same number of elements.
///
/// Emits nothing if the types match, a single collapse or expand where one
/// suffices, and otherwise a collapse to 1-D followed by an expand.
///
/// @retval nullptr   Shapes are dynamic or have different numbers of elements.
/// @retval Value     Value after reshaping.
[[nodiscard]] auto reshapeTo(OpBuilder& builder, Location loc,
                             TypedValue<RankedTensorType> value,
                             ArrayRef<int64_t> shape)
    -> TypedValue<RankedTensorType>;

/// Maps @p slice in @p source_shape to @p target_shape .
///
/// A copy transfer moves elements in order, so this mapping ignores unit
/// dimensions, but only works on statically known shapes.
///
/// @retval failure   Shapes are dynamic or differ in non-unit sizes.
/// @retval Slice     @p slice in @p target_shape .
[[nodiscard]] auto mapSliceOfCopy(Builder& builder, ArrayRef<Range> slice,
                                  ArrayRef<int64_t> source_shape,
                                  ArrayRef<int64_t> target_shape)
    -> FailureOr<SmallVector<Range>>;

/// Determines whether any op between @p begin and @p end has write effects.
///
/// If any op between @p begin and @p end might have a write side effect,
/// determined by recursive effect query, the result is `true`. The result is
/// also `true` if @p begin and @p end are not in the given order in the same
/// block. If the range is empty, the result is `false`.
[[nodiscard]] auto mightHaveWriteEffectsBetween(Block::iterator begin,
                                                Block::iterator end) -> bool;

/// Determines whether @p alloc slots might be affected by @p op .
///
/// If @p op might mutate the FifoResource in an unknown way, the result is
/// `true`. Otherwise, it is determined by the allocations of its slots. If an
/// allocation can't be determined, it is assumed to alias @p allocs .
[[nodiscard]] auto mightHaveSlotEffects(ArrayRef<FifoAllocateOp> allocs,
                                        Operation* op) -> bool;

/// Determines whether @p alloc slots are affected between @p begin and @p end .
///
/// See mightHaveSlotEffects(ArrayRef<FifoAllocateOp>, Operation*) for more
/// information.
[[nodiscard]] auto mightHaveSlotEffectsBetween(ArrayRef<FifoAllocateOp> allocs,
                                               Block::iterator begin,
                                               Block::iterator end) -> bool;

/// Adds @p patterns that coalesce data movement.
///
///  - Folds `tensor.(collapse|expand)_shape` into `ktdf.read_from_fifo`.
///  - Folds `tensor.(collapse|expand)_shape` into `ktdf.write_to_fifo`.
///  - Folds the producer into a `ktdf.read_from_fifo`.
///  - Folds two `ktdf.data_transfer` ops together.
///  - Folds `tensor.extract_slice` of a `ktdf.read_from_fifo` into the write.
///  - Folds `tensor.insert_slice` into a `ktdf.write_to_fifo` into the read.
///  - Folds `tensor.extract_slice` of a `ktdf.read_from_fifo` into the fill.
///  - Folds `tensor.insert_slice` into a `ktdf.write_to_fifo` into the drain.
///  - Folds `memref.subview` into a copy `ktdf.data_transfer`.
void populateDataMovementCoalescingPatterns(RewritePatternSet& patterns);

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_DATATRANSFERS_H_
