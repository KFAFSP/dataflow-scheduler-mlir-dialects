//===-- DataTransfers.cpp----------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/DataTransfers.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/ReshapeOpsUtils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/WalkResult.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

using MemRef = TypedValue<MemRefType>;
using Tensor = TypedValue<RankedTensorType>;

[[nodiscard]] auto getSingleUse(Value value) -> OpOperand* {
  const auto uses = value.getUses();
  if (uses.empty() || std::next(uses.begin()) != uses.end()) {
    return nullptr;
  }
  return &*uses.begin();
}

template <class OpType>
[[nodiscard]] auto getSingleUserOfType(Value value) -> OpType {
  auto* const use = getSingleUse(value);
  return use ? dyn_cast<OpType>(use->getOwner()) : nullptr;
}

struct ReshapeOp : mlir::Op<ReshapeOp> {
  using Op::Op;

  [[nodiscard]] static auto classof(Operation* op) -> bool {
    return isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op);
  }
};

/// Determines whether @p user can use @p value .
[[nodiscard]] auto isAvailable(Value value, Operation* user) -> bool {
  if (auto* const source = value.getDefiningOp(); source) {
    if (source->getBlock() == user->getBlock()) {
      return source->isBeforeInBlock(user);
    }
  }

  return value.getParentRegion()->isProperAncestor(user->getParentRegion());
}

/// Determines whether @p user can use @p mixed .
[[nodiscard]] auto areAvailable(ArrayRef<OpFoldResult> mixed, Operation* user)
    -> bool {
  return llvm::all_of(mixed, [&](OpFoldResult item) -> bool {
    const auto value = dyn_cast<Value>(item);
    return !value || isAvailable(value, user);
  });
}

/// Forwards @p mixed only if it is available at @p user .
[[nodiscard]] auto ensureAvailable(OpFoldResult mixed, Operation* user)
    -> OpFoldResult {
  if (!isa<Attribute>(mixed) && !isAvailable(cast<Value>(mixed), user)) {
    return nullptr;
  }
  return mixed;
}

/// Reshapes @p value to @p type , which must have the same number of elements.
///
/// Emits nothing if the types match, a single collapse or expand where one
/// suffices, and otherwise a collapse to 1-D followed by an expand.
///
/// @retval nullptr   Shapes are dynamic or have different numbers of elements.
/// @retval Value     Value after reshaping.
[[nodiscard]] auto reshapeTo(OpBuilder& builder, Location loc,
                             TypedValue<RankedTensorType> value,
                             RankedTensorType type) -> Value {
  const auto source_type = value.getType();
  if (source_type == type) {
    return value;
  }
  if (!source_type.hasStaticShape() || !type.hasStaticShape() ||
      source_type.getNumElements() != type.getNumElements()) {
    return nullptr;
  }

  if (const auto reassociation =
          getReassociationIndicesForReshape(source_type, type);
      reassociation) {
    if (source_type.getRank() > type.getRank()) {
      return tensor::CollapseShapeOp::create(builder, loc, type, value,
                                             *reassociation);
    }
    return tensor::ExpandShapeOp::create(builder, loc, type, value,
                                         *reassociation);
  }

  const auto flat_type =
      RankedTensorType::get({type.getNumElements()}, type.getElementType());
  Value flat = value;
  if (source_type.getRank() != 1) {
    flat = tensor::CollapseShapeOp::create(
        builder, loc, flat_type, value,
        {llvm::to_vector(llvm::seq<int64_t>(source_type.getRank()))});
  }
  if (type.getRank() == 1) {
    return flat;
  }
  return tensor::ExpandShapeOp::create(
      builder, loc, type, flat,
      {llvm::to_vector(llvm::seq<int64_t>(type.getRank()))});
}

/// Obtains the slice taken by @p op .
[[nodiscard]] auto getSlice(OffsetSizeAndStrideOpInterface op)
    -> SmallVector<Range> {
  return llvm::map_to_vector(
      llvm::zip_equal(op.getMixedOffsets(), op.getMixedSizes(),
                      op.getMixedStrides()),
      [](auto range) -> Range {
        auto [offset, size, stride] = range;
        return {offset, size, stride};
      });
}

void alwaysSucceeds(LogicalResult result) { assert(succeeded(result)); }

}  // namespace

//===----------------------------------------------------------------------===//
// mlir::ktdf::get(Producer|Consumer)
//===----------------------------------------------------------------------===//

auto mlir::ktdf::getProducer(FifoSlot slot) -> OpOperand* {
  OpOperand* result = nullptr;

  for (auto& use : getPrivateUses(slot)) {
    if (auto transfer = dyn_cast<DataTransferOp>(use.getOwner()); transfer) {
      if (&transfer.getDestinationMutable() != &use) {
        continue;
      }
    } else if (isa<ReadFromFifoOp>(use.getOwner())) {
      continue;
    } else if (!isa<WriteToFifoOp>(use.getOwner())) {
      return nullptr;
    }

    if (result) {
      return nullptr;
    }
    result = &use;
  }

  return result;
}

auto mlir::ktdf::getProducer(FifoSlot slot, Operation* user) -> OpOperand* {
  DominanceInfo dominance;
  auto* const producer = getProducer(slot);
  if (!producer ||
      producer->getOwner()->getParentOfType<StageOp>() !=
          user->getParentOfType<StageOp>() ||
      !dominance.dominates(producer->getOwner(), user)) {
    return nullptr;
  }

  return producer;
}

auto mlir::ktdf::getConsumer(FifoSlot slot) -> OpOperand* {
  OpOperand* result = nullptr;

  for (auto& use : getPrivateUses(slot)) {
    if (auto transfer = dyn_cast<DataTransferOp>(use.getOwner()); transfer) {
      if (&transfer.getSourceMutable() != &use) {
        continue;
      }
    } else if (isa<WriteToFifoOp>(use.getOwner())) {
      continue;
    } else if (!isa<ReadFromFifoOp>(use.getOwner())) {
      return nullptr;
    }

    if (result) {
      return nullptr;
    }
    result = &use;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// FifoSlotAllocation
//===----------------------------------------------------------------------===//

void FifoSlotAllocation::resize(RewriterBase& rewriter,
                                OpFoldResult num_elements) {
  rewriter.modifyOpInPlace(getOwner(), [&]() {
    getOwner().resize(getResultNumber(), num_elements);
  });

  for (auto& use : getUses()) {
    if (auto yield = dyn_cast<PrivateYieldOp>(use.getOwner()); yield) {
      yield->getParentOp()
          ->getResult(use.getOperandNumber())
          .setType(getType());
    }
  }
}

//===----------------------------------------------------------------------===//
// TransferAccess
//===----------------------------------------------------------------------===//

auto TransferAccess::get(MLIRContext* context, ArrayRef<Range> slice)
    -> FailureOr<TransferAccess> {
  TransferAccess result;

  // Each distinct value becomes one dimension.
  const auto get_dim = [&](Value value) -> AffineExpr {
    if (auto* const it = llvm::find(result.indices, value);
        it != result.indices.end()) {
      return getAffineDimExpr(std::distance(result.indices.begin(), it),
                              context);
    }
    result.indices.push_back(value);
    return getAffineDimExpr(result.indices.size() - 1, context);
  };

  SmallVector<AffineExpr> map_results;
  for (const auto& range : slice) {
    if (!isConstantIntValue(range.stride, 1) &&
        !isConstantIntValue(range.size, 1)) {
      // Slice is non-contiguous.
      return failure();
    }
    result.sizes.push_back(range.size);

    if (const auto value = getConstantIntValue(range.offset); value) {
      map_results.push_back(getAffineConstantExpr(*value, context));
    } else {
      map_results.push_back(get_dim(cast<Value>(range.offset)));
    }
  }

  result.base_map =
      AffineMap::get(result.indices.size(), 0, map_results, context);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::mapSliceOfCopy
//===----------------------------------------------------------------------===//

auto mlir::ktdf::mapSliceOfCopy(Builder& builder, ArrayRef<Range> slice,
                                ArrayRef<int64_t> source_shape,
                                ArrayRef<int64_t> target_shape)
    -> FailureOr<SmallVector<Range>> {
  assert(source_shape.size() == slice.size());

  SmallVector<Range> result;
  const Range unit{builder.getIndexAttr(0), builder.getIndexAttr(1),
                   builder.getIndexAttr(1)};
  const auto* source_it = source_shape.begin();
  const auto* slice_it = slice.begin();
  for (const auto target : target_shape) {
    if (ShapedType::isDynamic(target)) {
      return failure();
    }
    if (target == 1) {
      result.push_back(unit);
      continue;
    }
    while (true) {
      if (source_it == source_shape.end()) {
        return failure();
      }
      if (*source_it != 1) {
        break;
      }
      ++source_it;
      ++slice_it;
    }
    if (*source_it != target) {
      return failure();
    }
    result.push_back(*slice_it);
    ++source_it;
    ++slice_it;
  }
  if (llvm::any_of(llvm::make_range(source_it, source_shape.end()),
                   [](int64_t source) { return source != 1; })) {
    return failure();
  }

  return std::move(result);
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::mightHaveWriteEffectsBetween
//===----------------------------------------------------------------------===//

namespace {

template <class Pred>
[[nodiscard]] auto anyBetween(Block::iterator begin, Block::iterator end,
                              Pred pred) -> bool {
  const auto block_end = begin->getBlock()->end();
  for (auto it = std::next(begin); it != end; ++it) {
    if (it == block_end || pred(*it)) {
      return true;
    }
  }
  return false;
}

}  // namespace

auto mlir::ktdf::mightHaveWriteEffectsBetween(Block::iterator begin,
                                              Block::iterator end) -> bool {
  return anyBetween(begin, end, [](Operation& op) -> bool {
    const auto maybe_effects = getEffectsRecursively(&op);
    return !maybe_effects ||
           llvm::any_of(*maybe_effects,
                        [](const MemoryEffects::EffectInstance& effect) {
                          return isa<MemoryEffects::Write>(effect.getEffect());
                        });
  });
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::mightHaveSlotEffects(Between)
//===----------------------------------------------------------------------===//

auto mlir::ktdf::mightHaveSlotEffects(ArrayRef<FifoAllocateOp> allocs,
                                      Operation* op) -> bool {
  const auto visit = [&](Operation* op) -> WalkResult {
    const auto maybe_effects = getEffectsRecursively(op);
    if (!maybe_effects) {
      return WalkResult::interrupt();
    }
    if (maybe_effects->empty()) {
      return WalkResult::skip();
    }

    const auto might_share_alloc = [&](FifoSlot slot) -> bool {
      const auto def = getPrivateDefinition(slot);
      return !def || llvm::is_contained(allocs, def.getOwner());
    };

    return llvm::TypeSwitch<Operation*, WalkResult>(op)
        .Case([&](ReadFromFifoOp read) -> WalkResult {
          return might_share_alloc(read.getFifoSlot()) ? WalkResult::interrupt()
                                                       : WalkResult::advance();
        })
        .Case([&](WriteToFifoOp write) -> WalkResult {
          return might_share_alloc(write.getFifoSlot())
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .Case([&](DataTransferOp transfer) -> WalkResult {
          if (auto source = dyn_cast<FifoSlot>(transfer.getSource());
              source && might_share_alloc(source)) {
            return WalkResult::interrupt();
          }
          if (auto dest = dyn_cast<FifoSlot>(transfer.getDestination());
              dest && might_share_alloc(dest)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .Default([&](Operation* /*op*/) -> WalkResult {
          for (const auto& effect : *maybe_effects) {
            if (isa<MemoryEffects::Write>(effect.getEffect()) &&
                isa<FifoResource>(effect.getResource())) {
              return WalkResult::interrupt();
            }
          }

          return WalkResult::advance();
        });
  };

  return op->walk<WalkOrder::PreOrder>(visit).wasInterrupted();
}

auto mlir::ktdf::mightHaveSlotEffectsBetween(ArrayRef<FifoAllocateOp> allocs,
                                             Block::iterator begin,
                                             Block::iterator end) -> bool {
  return anyBetween(begin, end, [&](Operation& op) -> bool {
    return mightHaveSlotEffects(allocs, &op);
  });
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::populateDataMovementCoalescingPatterns
//===----------------------------------------------------------------------===//

namespace {

/// Folds a reshape into a `ktdf.read_from_fifo`.
struct FoldReshapeIntoFifoRead : OpRewritePattern<ReadFromFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(ReadFromFifoOp read, PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto reshape = getSingleUserOfType<ReshapeOp>(read);
    if (!reshape) {
      return rewriter.notifyMatchFailure(read, "not reshaped");
    }

    rewriter.modifyOpInPlace(read, [&]() {
      read.getResult().setType(
          cast<ShapedType>(reshape->getResult(0).getType()));
    });
    rewriter.replaceOp(reshape, read.getResult());
    return success();
  }
};

/// Folds a reshape of into a `ktdf.write_to_fifo`.
struct FoldReshapeIntoFifoWrite : OpRewritePattern<WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(WriteToFifoOp write, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto reshape = write.getData().getDefiningOp<ReshapeOp>();
    if (!reshape) {
      return rewriter.notifyMatchFailure(write, "not reshaped");
    }

    rewriter.modifyOpInPlace(write, [&]() {
      write.getDataMutable().assign(reshape->getOperand(0));
    });
    return success();
  }
};

/// Folds the producer into a `ktdf.read_from_fifo`.
struct FoldProducerIntoFifoRead : OpRewritePattern<ReadFromFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(ReadFromFifoOp read, PatternRewriter& rewriter) const
      -> LogicalResult override {
    const auto result_type = dyn_cast<RankedTensorType>(read.getType());
    if (!result_type || read.getSplat()) {
      return rewriter.notifyMatchFailure(read,
                                         "requires non-splat tensor semantics");
    }
    auto* const producer = getProducer(read.getFifoSlot(), read);
    if (!producer) {
      return rewriter.notifyMatchFailure(read, "no producer in the same stage");
    }

    // Handle `ktdf.write_to_fifo` producers with tensor data.
    if (auto write = dyn_cast<WriteToFifoOp>(producer->getOwner()); write) {
      const auto data = dyn_cast<Tensor>(write.getData());
      if (!data) {
        return rewriter.notifyMatchFailure(write, "requires tensor data");
      }

      // There might be reshaping between these operations.
      const auto reshaped =
          reshapeTo(rewriter, read.getLoc(), data, result_type);
      if (!reshaped) {
        return rewriter.notifyMatchFailure(read, "dynamic reshape");
      }

      // The `ktdf.write_to_fifo` can be omitted and the slot becomes dead.
      rewriter.replaceOp(read, reshaped);
      rewriter.eraseOp(write);
      return success();
    }

    // Handle `ktdf.data_transfer` producers with FIFO sources.
    auto transfer = cast<DataTransferOp>(producer->getOwner());
    if (const auto source_alloc = FifoSlotAllocation::get(transfer.getSource());
        source_alloc) {
      // There is no offset into a FIFO, so only whole transfers can be read.
      const auto sizes = transfer.getMixedSourceSizes();
      const auto slot_size = ensureAvailable(source_alloc.getMixedSize(), read);
      if (sizes.size() != 1 || !slot_size ||
          !isEqualConstantIntOrValue(sizes.front(), slot_size)) {
        return rewriter.notifyMatchFailure(
            transfer, "partial or unsizeable FIFO transfer");
      }
      if (mightHaveSlotEffectsBetween({source_alloc.getOwner()},
                                      transfer->getIterator(),
                                      read->getIterator())) {
        return rewriter.notifyMatchFailure(read, "slot access order changes");
      }

      // The `ktdf.data_transfer` can be replaced by a `ktdf.read_from_fifo`.
      rewriter.modifyOpInPlace(read, [&]() {
        read.getFifoSlotMutable().set(transfer.getSource());
        read->setLoc(rewriter.getFusedLoc({transfer.getLoc(), read.getLoc()}));
      });
      rewriter.eraseOp(transfer);
      return success();
    }

    // We're unable to insert a tensorized load, or don't know the allocation.
    return rewriter.notifyMatchFailure(transfer,
                                       "memory or unallocated FIFO source");
  }
};

/// Folds two `ktdf.data_transfer` ops together.
struct FoldDataTransferRelay : OpRewritePattern<DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(DataTransferOp transfer, PatternRewriter& rewriter) const
      -> LogicalResult override {
    const auto dest_alloc = FifoSlotAllocation::get(transfer.getSource());
    if (transfer.getTransferMode() != TransferMode::Copy || !dest_alloc) {
      return rewriter.notifyMatchFailure(
          transfer, "requires copy from source FIFO with known allocation");
    }
    auto* const producer = getProducer(dest_alloc, transfer);
    if (!producer) {
      return rewriter.notifyMatchFailure(transfer,
                                         "no producer in the same stage");
    }
    auto hop = dyn_cast<DataTransferOp>(producer->getOwner());
    if (!hop) {
      return rewriter.notifyMatchFailure(producer->getOwner(),
                                         "unable to fold FIFO write");
    }
    if (hop.getTransferMode() != TransferMode::Copy) {
      return rewriter.notifyMatchFailure(hop,
                                         "hop does not have copy semantics");
    }
    SmallVector<FifoAllocateOp> allocs{dest_alloc.getOwner()};
    if (const auto source_alloc = FifoSlotAllocation::get(hop.getSource());
        source_alloc) {
      allocs.push_back(source_alloc.getOwner());
    } else if (mightHaveWriteEffectsBetween(hop->getIterator(),
                                            transfer->getIterator())) {
      return rewriter.notifyMatchFailure(transfer,
                                         "potential buffer modification");
    }
    if (mightHaveSlotEffectsBetween(allocs, hop->getIterator(),
                                    transfer->getIterator())) {
      return rewriter.notifyMatchFailure(transfer, "FIFO access order changes");
    }

    rewriter.modifyOpInPlace(transfer, [&]() {
      transfer.getSourceMutable().set(hop.getSource());
      transfer.setSourceMapAttr(hop.getSourceMapAttr());
      transfer.getSourceIndicesMutable().assign(hop.getSourceIndices());
      transfer.getSourceSizesMutable().assign(hop.getSourceSizes());
      transfer.setStaticSourceSizesAttr(hop.getStaticSourceSizesAttr());
      transfer->setLoc(rewriter.getFusedLoc({hop.getLoc(), transfer.getLoc()}));
    });
    rewriter.eraseOp(hop);
    return success();
  }
};

/// Folds a `tensor.extract_slice` of a `ktdf.read_from_fifo` into the write.
struct FoldExtractSliceIntoFifoRead : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(tensor::ExtractSliceOp extract,
                       PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto read = extract.getSource().getDefiningOp<ReadFromFifoOp>();
    if (!read || !read->hasOneUse()) {
      return rewriter.notifyMatchFailure(extract, "not the only use of a read");
    }
    if (read.getSplat()) {
      return rewriter.notifyMatchFailure(read, "read splats");
    }
    if (!extract.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(extract, "dynamic slice");
    }
    auto alloc = FifoSlotAllocation::get(read.getFifoSlot());
    if (!alloc || alloc.getConsumer() != &read.getFifoSlotMutable()) {
      return rewriter.notifyMatchFailure(read, "unknown slot allocation");
    }
    auto* const producer = alloc.getProducer();
    auto write =
        producer ? dyn_cast<WriteToFifoOp>(producer->getOwner()) : nullptr;
    if (!write || write.getData().getType() != read.getType()) {
      return rewriter.notifyMatchFailure(read, "not written as read");
    }
    if (!areAvailable(extract.getMixedOffsets(), write) ||
        !areAvailable(extract.getMixedSizes(), write) ||
        !areAvailable(extract.getMixedStrides(), write)) {
      return rewriter.notifyMatchFailure(extract, "slice not available");
    }

    // The slice moves to the write, and the read takes its place.
    rewriter.replaceAllUsesWith(extract.getResult(), read.getResult());
    rewriter.moveOpBefore(extract, write);
    rewriter.modifyOpInPlace(
        extract, [&]() { extract.getSourceMutable().assign(write.getData()); });
    rewriter.modifyOpInPlace(
        write, [&]() { write.getDataMutable().assign(extract.getResult()); });
    alloc.resize(rewriter,
                 rewriter.getIndexAttr(extract.getType().getNumElements()));
    rewriter.modifyOpInPlace(
        read, [&]() { read.getResult().setType(extract.getType()); });
    return success();
  }
};

/// Folds a `tensor.insert_slice` into a `tensor.empty` into the read.
struct FoldInsertSliceIntoFifoWrite : OpRewritePattern<WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(WriteToFifoOp write, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto insert = write.getData().getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || !insert->hasOneUse() ||
        !insert.getDest().getDefiningOp<tensor::EmptyOp>()) {
      return rewriter.notifyMatchFailure(write, "not a slice into an empty");
    }
    if (!insert.getSourceType().hasStaticShape() ||
        !insert.getDestType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(insert, "dynamic slice");
    }
    auto alloc = FifoSlotAllocation::get(write.getFifoSlot());
    if (!alloc || alloc.getProducer() != &write.getFifoSlotMutable()) {
      return rewriter.notifyMatchFailure(write, "unknown slot allocation");
    }
    auto* const consumer = alloc.getConsumer();
    auto read =
        consumer ? dyn_cast<ReadFromFifoOp>(consumer->getOwner()) : nullptr;
    if (!read || read.getSplat() || read.getType() != insert.getDestType()) {
      return rewriter.notifyMatchFailure(write, "not read as written");
    }
    if (!areAvailable(insert.getMixedOffsets(), read) ||
        !areAvailable(insert.getMixedSizes(), read) ||
        !areAvailable(insert.getMixedStrides(), read)) {
      return rewriter.notifyMatchFailure(insert, "slice not available");
    }

    // The write takes the slice, and the slice moves to the read.
    rewriter.modifyOpInPlace(
        write, [&]() { write.getDataMutable().assign(insert.getSource()); });
    alloc.resize(rewriter, rewriter.getIndexAttr(
                               insert.getSourceType().getNumElements()));
    rewriter.modifyOpInPlace(
        read, [&]() { read.getResult().setType(insert.getSourceType()); });
    rewriter.setInsertionPointAfter(read);
    auto empty = tensor::EmptyOp::create(rewriter, insert.getLoc(),
                                         insert.getDestType().getShape(),
                                         insert.getDestType().getElementType());
    rewriter.moveOpAfter(insert, empty);
    rewriter.replaceAllUsesWith(read.getResult(), insert.getResult());
    rewriter.modifyOpInPlace(insert, [&]() {
      insert.getSourceMutable().assign(read.getResult());
      insert.getDestMutable().assign(empty);
    });
    return success();
  }
};

/// Folds a `memref.subview` into the memory side of a copy.
struct FoldSubViewIntoTransfer : OpRewritePattern<DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(DataTransferOp transfer, PatternRewriter& rewriter) const
      -> LogicalResult override {
    switch (transfer.getTransferMode()) {
      case TransferMode::Copy:
        break;

      case TransferMode::Splat:
      case TransferMode::ZeroPadFront:
      case TransferMode::ZeroPadBack:
        return rewriter.notifyMatchFailure(transfer, "not a copy");
    }

    rewriter.startOpModification(transfer);
    const auto source = foldSide(rewriter, transfer.getSourceMutable());
    const auto dest = foldSide(rewriter, transfer.getDestinationMutable());

    if (failed(source) && failed(dest)) {
      rewriter.cancelOpModification(transfer);
      return rewriter.notifyMatchFailure(transfer, "no foldable subview");
    }

    rewriter.finalizeOpModification(transfer);
    return success();
  }

  /// Attempts to fold @p slice into a memory @p side of a copy transfer.
  ///
  /// @pre  `isa<MemRef>(side.get())`
  /// @pre  `isa<DataTransferOp>(side.getOwner())`
  /// @pre  `transfer.getTransferMode() == TransferMode::Copy`
  [[nodiscard]] static auto foldSide(OpBuilder& builder, OpOperand& side,
                                     ArrayRef<Range> view) -> LogicalResult {
    auto transfer = cast<DataTransferOp>(side.getOwner());
    assert(transfer.getTransferMode() == TransferMode::Copy);
    assert(isa<MemRef>(side.get()));

    // The merged access must stay contiguous. Checked before building any IR.
    const auto sizes = transfer.getTiedSizes(&side);
    for (auto [size, range] : llvm::zip_equal(sizes, view)) {
      if (!isConstantIntValue(range.stride, 1) &&
          !isConstantIntValue(size, 1)) {
        return failure();
      }
    }

    // The transfer reads its sizes contiguously from its offsets, which the
    // view places at its offsets and scales by its strides.
    const auto [base_map, indices] = transfer.getTiedOffset(&side);
    const auto offsets = affine::makeComposedFoldedMultiResultAffineApply(
        builder, transfer.getLoc(), base_map, getAsOpFoldResult(indices));
    AffineExpr view_offset, view_stride, offset;
    bindSymbols(builder.getContext(), view_offset, view_stride, offset);
    const auto slice = llvm::map_to_vector(
        llvm::zip_equal(view, offsets, sizes), [&](auto dim) -> Range {
          auto [range, offset_value, size] = dim;
          return {affine::makeComposedFoldedAffineApply(
                      builder, transfer.getLoc(),
                      view_offset + view_stride * offset,
                      {range.offset, range.stride, offset_value}),
                  size, range.stride};
        });
    auto access = TransferAccess::get(builder.getContext(), slice);
    assert(succeeded(access) && "contiguity was checked above");

    transfer.setTiedOffset(&side, access->base_map, access->indices);
    return success();
  }

  /// Narrows a copying fill or drain of a FIFO from @p buffer_side .
  ///
  /// @pre  `isa<MemRef>(side.get())`
  /// @pre  `isa<DataTransferOp>(side.getOwner())`
  /// @pre  `transfer.getTransferMode() == TransferMode::Copy`
  [[nodiscard]] static auto narrowCopyFillDrain(OpBuilder& builder,
                                                OpOperand& buffer_side,
                                                ArrayRef<Range> slice,
                                                ArrayRef<int64_t> slice_shape)
      -> LogicalResult {
    auto transfer = cast<DataTransferOp>(buffer_side.getOwner());
    assert(transfer.getTransferMode() == TransferMode::Copy);
    assert(isa<MemRef>(buffer_side.get()));

    const auto is_source = &buffer_side == &transfer.getSourceMutable();
    const auto region =
        getConstantIntValues(transfer.getTiedSizes(&buffer_side));
    if (!region) {
      return failure();
    }
    const auto mapped = mapSliceOfCopy(builder, slice, slice_shape, *region);
    if (failed(mapped)) {
      return failure();
    }

    SmallVector<int64_t> sizes;
    int64_t num_elements = 1;
    for (const auto range : *mapped) {
      const auto size = getConstantIntValue(range.size);
      if (!size || (!isConstantIntValue(range.stride, 1) && size != 1) ||
          !areAvailable({range.offset, range.stride}, transfer)) {
        return failure();
      }

      sizes.push_back(*size);
      num_elements *= *size;
    }

    if (is_source) {
      transfer.setStaticSourceSizes(sizes);
      transfer.setStaticDestSizes({num_elements});
      alwaysSucceeds(foldSide(builder, transfer.getSourceMutable(), *mapped));
    } else {
      transfer.setStaticDestSizes(sizes);
      transfer.setStaticSourceSizes({num_elements});
      alwaysSucceeds(
          foldSide(builder, transfer.getDestinationMutable(), *mapped));
    }

    return success();
  }

 private:
  [[nodiscard]] static auto foldSide(PatternRewriter& rewriter, OpOperand& side)
      -> LogicalResult {
    auto subview = side.get().getDefiningOp<memref::SubViewOp>();
    if (!isa<MemRef>(side.get()) || !subview ||
        subview.getSourceType().getRank() != subview.getType().getRank() ||
        failed(foldSide(rewriter, side, getSlice(subview)))) {
      return failure();
    }

    side.assign(subview.getSource());
    if (subview->use_empty()) {
      rewriter.eraseOp(subview);
    }
    return success();
  }
};

/// Folds `tensor.extract_slice` of a `ktdf.read_from_fifo` into the fill.
struct FoldExtractSliceIntoFifoFill : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(tensor::ExtractSliceOp extract,
                       PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto read = extract.getSource().getDefiningOp<ReadFromFifoOp>();
    if (!read || !read->hasOneUse() || read.getSplat() ||
        !extract.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(extract,
                                         "not a static slice of a read");
    }
    auto alloc = FifoSlotAllocation::get(read.getFifoSlot());
    auto* const producer = alloc ? alloc.getProducer() : nullptr;
    auto transfer =
        producer ? dyn_cast<DataTransferOp>(producer->getOwner()) : nullptr;
    if (!transfer || !isa<MemRef>(transfer.getSource()) ||
        transfer.getTransferMode() != TransferMode::Copy) {
      return rewriter.notifyMatchFailure(read, "not filled by a copy");
    }
    if (failed(FoldSubViewIntoTransfer::narrowCopyFillDrain(
            rewriter, transfer.getSourceMutable(), getSlice(extract),
            extract.getSourceType().getShape()))) {
      return rewriter.notifyMatchFailure(extract, "not a slice of the copy");
    }

    alloc.resize(rewriter,
                 rewriter.getIndexAttr(extract.getType().getNumElements()));
    rewriter.modifyOpInPlace(
        read, [&]() { read.getResult().setType(extract.getType()); });
    rewriter.replaceOp(extract, read.getResult());
    return success();
  }
};

/// Folds `tensor.insert_slice` into a `ktdf.write_to_fifo` into the drain.
struct FoldInsertSliceIntoFifoDrain : OpRewritePattern<WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(WriteToFifoOp write, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto insert = write.getData().getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || !insert->hasOneUse() ||
        !insert.getDest().getDefiningOp<tensor::EmptyOp>() ||
        !insert.getSourceType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(write,
                                         "not a static slice of an empty");
    }
    auto alloc = FifoSlotAllocation::get(write.getFifoSlot());
    auto* const consumer = alloc ? alloc.getConsumer() : nullptr;
    auto transfer =
        consumer ? dyn_cast<DataTransferOp>(consumer->getOwner()) : nullptr;
    if (!transfer || !isa<MemRef>(transfer.getDestination()) ||
        transfer.getTransferMode() != TransferMode::Copy) {
      return rewriter.notifyMatchFailure(write, "not drained by a copy");
    }
    if (failed(FoldSubViewIntoTransfer::narrowCopyFillDrain(
            rewriter, transfer.getDestinationMutable(), getSlice(insert),
            insert.getDestType().getShape()))) {
      return rewriter.notifyMatchFailure(insert, "not a slice of the copy");
    }

    alloc.resize(rewriter, rewriter.getIndexAttr(
                               insert.getSourceType().getNumElements()));
    rewriter.modifyOpInPlace(
        write, [&]() { write.getDataMutable().assign(insert.getSource()); });
    rewriter.eraseOp(insert);
    return success();
  }
};

}  // namespace

void mlir::ktdf::populateDataMovementCoalescingPatterns(
    RewritePatternSet& patterns) {
  patterns.add<FoldReshapeIntoFifoRead, FoldReshapeIntoFifoWrite,
               FoldProducerIntoFifoRead, FoldDataTransferRelay,
               FoldExtractSliceIntoFifoRead, FoldInsertSliceIntoFifoWrite,
               FoldSubViewIntoTransfer, FoldExtractSliceIntoFifoFill,
               FoldInsertSliceIntoFifoDrain>(patterns.getContext());
  DataTransferOp::getCanonicalizationPatterns(patterns, patterns.getContext());
}
