//===-- PipelineBuilder.cpp -------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageDependency.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

#define DEBUG_TYPE "ktdf-pipeline-builder"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

const auto kSkipRegions = OpPrintingFlags().skipRegions();

}  // namespace

//===----------------------------------------------------------------------===//
// PipelineBuilder::Allocator
//===----------------------------------------------------------------------===//

namespace {

[[nodiscard]] auto getUnitOrUnits(StageOp stage) -> Attribute {
  const auto units = stage.getApplicableUnitsAttr();
  if (!units) {
    return ArrayAttr::get(stage->getContext(), {});
  }

  return units.size() == 1 ? units.getValue().front() : units;
}

}  // namespace

auto PipelineBuilder::Allocator::getDefault() -> Allocator& {
  static Allocator instance;
  return instance;
}

auto PipelineBuilder::Allocator::canAllocate(OpResult producer) const -> bool {
  const auto value_type = dyn_cast<ShapedType>(producer.getType());
  return value_type && !isa<MemRefType>(value_type) &&
         value_type.hasStaticShape();
}

auto PipelineBuilder::Allocator::allocate(PipelineBuilder& builder,
                                          OpResult producer, StageOp consumer)
    -> TypedValue<FifoSlotType> {
  const auto value_type = cast<ShapedType>(producer.getType());

  // We can't just read from an existing FIFO slot, we'll have to make a copy
  // of the value at the producer's location and plumb a new FIFO.
  const auto producer_unit =
      getUnitOrUnits(producer.getOwner()->getParentOfType<StageOp>());
  const auto consumer_unit = getUnitOrUnits(consumer);

  // Create the appropriate type for the slot, which flattens the elements.
  const auto slot_type = FifoSlotType::get(
      builder.getContext(), producer_unit, consumer_unit,
      value_type.getNumElements(), value_type.getElementType());
  return cast<TypedValue<FifoSlotType>>(
      builder.getPrivateBuilder()
          .createFifo({slot_type}, {}, consumer->getLoc())
          .front());
}

//===----------------------------------------------------------------------===//
// PipelineBuilder
//===----------------------------------------------------------------------===//

auto PipelineBuilder::getUnits(Attribute attr) const -> ArrayAttr {
  auto units = dyn_cast_if_present<ArrayAttr>(attr);
  if (!units && attr) {
    units = ArrayAttr::get(getContext(), {attr});
  }

  return units;
}

auto PipelineBuilder::getReadBuilder(StageOp stage) -> OpBuilder {
  auto& body = *stage.getBody();
  auto it = body.begin();
  while (it != body.end() && isa<ReadFromFifoOp>(&*it)) {
    ++it;
  }

  return OpBuilder(stage.getBody(), it, listener);
}

auto PipelineBuilder::getWriteBuilder(StageOp stage) -> OpBuilder {
  auto& body = *stage.getBody();

  return OpBuilder(stage.getBody(), body.end(), listener);
}

auto PipelineBuilder::createStage(ArrayAttr units, BodyBuilderFn body_builder,
                                  std::optional<Location> loc) -> StageOp {
  auto result =
      StageOp::create(*this, loc.value_or(getLoc()), {}, {}, body_builder);
  if (units) {
    result.setApplicableUnitsAttr(units);
    units_to_stage_[units] = result;
  }

  OpBuilder::setInsertionPoint(result);
  return result;
}

auto PipelineBuilder::addDependency(StageOp producer, StageOp consumer)
    -> bool {
  if (producer->getBlock() != getInsertionBlock() ||
      consumer->getBlock() != getInsertionBlock()) {
    return false;
  }

  IRRewriter rewriter(*this);
  auto token = tokens_.lookup(producer);
  if (!token) {
    tokens_[producer] = token =
        private_builder_.createToken(producer->getLoc());
    rewriter.modifyOpInPlace(producer,
                             [&]() { producer.addOutDependency(token); });
  }

  rewriter.startOpModification(consumer);
  if (!consumer.addInDependency(token)) {
    rewriter.cancelOpModification(consumer);
    return false;
  }
  rewriter.finalizeOpModification(consumer);
  dependencies_.insert(consumer, producer);
  return true;
}

auto PipelineBuilder::addDependency(Operation* producer, Operation* consumer)
    -> bool {
  auto producer_stage = producer->getParentOfType<StageOp>();
  auto consumer_stage = consumer->getParentOfType<StageOp>();
  if (!producer_stage || !consumer_stage) {
    return false;
  }

  return addDependency(producer_stage, consumer_stage);
}

auto PipelineBuilder::isAvailable(OpResult producer, StageOp consumer) const
    -> bool {
  return producer.getParentRegion()->isAncestor(consumer->getParentRegion()) ||
         private_builder_.isPrivate(producer.getOwner());
}

auto PipelineBuilder::canForward(OpResult producer) const -> bool {
  auto stage = producer.getOwner()->getParentOfType<StageOp>();
  return stage && stage->getBlock() == getInsertionBlock() &&
         allocator_->canAllocate(producer);
}

auto PipelineBuilder::forward(OpResult producer, StageOp consumer) -> Value {
  if (isAvailable(producer, consumer)) {
    // The consumer already has access to the value, either because the producer
    // is in the consumer stage, the private segment, or outside the pipeline.
    return producer;
  }

  assert(canForward(producer));

  auto producer_stage = producer.getOwner()->getParentOfType<StageOp>();
  assert(producer_stage && producer_stage->getBlock() == getInsertionBlock());

  // Lookup existing FIFO reads that produce this value.
  auto& reads = fifos_[producer];
  for (auto read : reads) {
    if (read->getParentOp() == consumer) {
      return read;
    }
  }

  // Allocate a new FIFO slot for this value.
  const auto slot = allocator_->allocate(*this, producer, consumer);
  assert(slot && "allocator may not fail");

  // On the producer side, create a new write to the slot.
  {
    auto builder = getWriteBuilder(producer_stage);
    WriteToFifoOp::create(builder, consumer->getLoc(), producer, slot);
  }

  // On the consumer side, create a new read from the slot.
  ReadFromFifoOp read;
  {
    auto builder = getReadBuilder(consumer);
    read = ReadFromFifoOp::create(builder, consumer->getLoc(),
                                  producer.getType(), slot);
    reads.push_back(read);
  }

  // Introduce a dependency between producer and consumer via private tokens.
  addDependency(producer_stage, consumer);
  return read;
}

auto PipelineBuilder::naturalPlacement(Operation* op) -> Placement {
  StageOp result;
  for (auto* const user : op->getUsers()) {
    auto consumer_stage = user->getParentOfType<StageOp>();
    if (!consumer_stage || (result && consumer_stage != result)) {
      return nullptr;
    }

    result = consumer_stage;
  }

  return result;
}

namespace {

enum class ForwardingResult : char {
  Failure = 0,
  Forward = 1,
  SplitAndForward = 2,
  Skip = 3,
};

[[nodiscard]] auto checkResult(PipelineBuilder& builder, StageOp stage,
                               OpResult result) -> ForwardingResult {
  auto status = ForwardingResult::Skip;

  for (auto* const user : result.getUsers()) {
    auto consumer_stage = user->getParentOfType<StageOp>();
    assert(consumer_stage);
    if (consumer_stage == stage) {
      // This results stays within the placement stage, so doesn't need to
      // be forwarded.
      continue;
    }

    if (builder.hasDependency(consumer_stage, stage)) {
      // Forwarding this result would create a cyclic dependency.
      LDBG() << "  (WARN) detected dependency cycle between";
      LDBG() << "    consumer: " << OpWithFlags(consumer_stage, kSkipRegions);
      LDBG() << "    producer: " << OpWithFlags(stage, kSkipRegions);

      // We can break this cycle by creating a new stage. This stage will
      // copy the units of the old stage, and take its place in the map.
      status = ForwardingResult::SplitAndForward;
      break;
    }

    status = ForwardingResult::Forward;
  }

  if (status == ForwardingResult::Skip) {
    // We don't have to forward this result.
    return ForwardingResult::Skip;
  }

  if (!builder.getAllocator().canAllocate(result)) {
    LDBG() << "  (FAILED) result #" << result.getResultNumber()
           << " can't be forwarded";
    return ForwardingResult::Failure;
  }

  return ForwardingResult::Forward;
};

[[nodiscard]] auto checkResults(PipelineBuilder& builder, StageOp placement,
                                Operation* op,
                                SmallVectorImpl<OpResult>& forward)
    -> ForwardingResult {
  auto status = ForwardingResult::Forward;
  for (auto result : op->getResults()) {
    switch (checkResult(builder, placement, result)) {
      case ForwardingResult::Failure:
        return ForwardingResult::Failure;
      case ForwardingResult::Forward:
        break;
      case ForwardingResult::SplitAndForward:
        status = ForwardingResult::SplitAndForward;
        break;
      case ForwardingResult::Skip:
        continue;
    }

    forward.push_back(result);
  }

  return status;
}

}  // namespace

auto PipelineBuilder::insert(Operation* op, Placement placement)
    -> LogicalResult {
  assert(placement);

  LDBG() << "trying to insert";
  LDBG() << "    op: " << OpWithFlags(op, kSkipRegions);
  LDBG() << "  into: " << OpWithFlags(placement.stage, kSkipRegions);

  const auto is_in_pipeline = [&](Operation* op) -> bool {
    return getInsertionBlock()->getParentOp()->isAncestor(op) ||
           private_builder_.isPrivate(op);
  };
  if (is_in_pipeline(op)) {
    // This operation is already in the pipeline.
    LDBG() << "  (FAILED) already inside pipeline";
    return failure();
  }
  if (!llvm::all_of(op->getUsers(), is_in_pipeline)) {
    // This operation is already in the pipeline.
    LDBG() << "  (FAILED) has users outside of pipeline";
    return failure();
  }

  // Determine all the results that we will have to forward, checking for
  // dependency cycles in the process.
  SmallVector<OpResult> results_to_forward;
  switch (checkResults(*this, placement.stage, op, results_to_forward)) {
    case ForwardingResult::Failure:
      return failure();
    case ForwardingResult::Forward:
      break;
    case ForwardingResult::SplitAndForward: {
      // We can insert the op, but we have to break a dependency cycle. We can
      // do this by creating a new stage from the desired placement.
      auto old_stage = std::exchange(
          placement.stage, createStage(placement.stage.getApplicableUnitsAttr(),
                                       {}, placement.stage.getLoc()));
      if (placement.erase_on_failure) {
        // This counts as a failure.
        erase(old_stage);
      }
      break;
    }
    case ForwardingResult::Skip:
      llvm_unreachable("unexpected ForwardingResult");
  }

  // This op can safely be moved into the pipeline.
  {
    op->remove();
    getReadBuilder(placement.stage).insert(op);
  }

  // Forward all results of this operation to their in-pipeline users.
  for (auto result : results_to_forward) {
    for (auto& use : result.getUses()) {
      auto stage = use.getOwner()->getParentOfType<StageOp>();
      assert(stage && "user outside of pipeline");
      auto forwarded = forward(result, stage);
      use.set(forwarded);
    }
  }

  LDBG() << "  (SUCCESS) inserted";
  return success();
}

namespace {

void dominanceSort(MutableArrayRef<Operation*> op, DominanceInfo& dominance) {
  llvm::stable_sort(op, [&](Operation* lhs, Operation* rhs) -> bool {
    // NOTE: This code will not work for graph regions, and there is no simple
    //       way to fix that. However, that's not an intended usage scenario.
    return dominance.properlyDominates(lhs, rhs);
  });
}

}  // namespace

void PipelineBuilder::insert(ArrayRef<Operation*> ops, PlacementFn placement_fn,
                             DominanceInfo& dominance) {
  // Initialize the work list in reverse order, since we're popping from the
  // back and want to keep the order (to preserve SSA property).
  llvm::SmallVector<Operation*> work_list(ops.rbegin(), ops.rend());
  while (!work_list.empty()) {
    auto* const op = work_list.pop_back_val();
    const auto placement = placement_fn(*this, op);
    if (!placement || failed(insert(op, placement))) {
      continue;
    }

    // Consider all the inserted ops producers for insertion into the pipeline.
    const auto split = work_list.size();
    for (auto needs : op->getOperands()) {
      if (const auto wants = dyn_cast<OpResult>(needs); wants) {
        LDBG() << "inserted op wants "
               << OpWithFlags(wants.getOwner(), kSkipRegions);
        work_list.push_back(wants.getOwner());
      }
    }
    // We need to ensure that dependencies between the producers do not prevent
    // them from being inserted into the pipeline. We can prevent this from
    // happening by visiting the producers in reverse dominance order.
    dominanceSort(MutableArrayRef(work_list.data() + split, work_list.end()),
                  dominance);
  }
}

auto PipelineBuilder::build() -> PipelineOp {
  // Erase all the empty stages back to front (to erase forwarding chains).
  if (!getInsertionBlock()->empty()) {
    auto* it = &getInsertionBlock()->back();
    do {
      auto stage = dyn_cast<StageOp>(it);
      it = it->getPrevNode();
      if (stage && stage.getBody()->empty() && stage.getDependsOut().empty()) {
        erase(stage);
      }
    } while (it != nullptr);
  }

  // Build the PrivateOp and erase it if it's empty.
  if (auto private_op = private_builder_.build();
      private_op.getBody()->without_terminator().empty()) {
    IRRewriter(*this).eraseOp(private_op);
  }

  return cast<PipelineOp>(getInsertionBlock()->getParentOp());
}

PipelineBuilder::PipelineBuilder(PipelineOp pipeline, Allocator* allocator,
                                 OpBuilder::Listener* listener)
    : ImplicitLocOpBuilder(pipeline.getLoc(), pipeline.getContext()),
      private_builder_(pipeline, std::nullopt, listener),
      allocator_(allocator != nullptr ? allocator : &Allocator::getDefault()) {
  OpBuilder::setInsertionPointToEnd(pipeline.getBody());
}

namespace {

[[nodiscard]] auto getSingleUser(Value value) -> Operation* {
  const auto users = value.getUsers();
  if (users.empty() || std::next(users.begin()) != users.end()) {
    return nullptr;
  }
  return *users.begin();
}

template <class OpType>
[[nodiscard]] auto getSingleUserOfType(Value value) -> OpType {
  return dyn_cast_if_present<OpType>(getSingleUser(value));
}

}  // namespace

void PipelineBuilder::erase(StageOp stage) {
  // Erase the insertion point mapping.
  // TODO: Figure out a new one?
  if (const auto it = units_to_stage_.find(stage.getApplicableUnitsAttr());
      it != units_to_stage_.end() && it->second == stage) {
    units_to_stage_.erase(it);
  }

  // Erase the dependency information.
  // TODO: Remove the token from all the consumers, potentially making it dead.
  tokens_.erase(stage);
  dependencies_.erase(stage);

  // Erase the FIFO reads we know about in this stage.
  IRRewriter rewriter(*this);
  const auto erase_fifo = [&](ReadFromFifoOp read) {
    read->dropAllReferences();
    if (auto write = getSingleUserOfType<WriteToFifoOp>(read.getFifoSlot());
        write) {
      rewriter.eraseOp(write);
    }
    rewriter.eraseOp(read);
  };
  for (auto&& [source, reads] : fifos_) {
    llvm::erase_if(reads, [&](ReadFromFifoOp read) -> bool {
      if (read->getParentOp() == stage) {
        erase_fifo(read);
        return true;
      }

      return false;
    });
  }

  rewriter.eraseOp(stage);
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::eliminiateVia
//===----------------------------------------------------------------------===//

auto mlir::ktdf::eliminateVia(RewriterBase& rewriter, ViaOp via)
    -> LogicalResult {
  auto write = getSingleUserOfType<WriteToFifoOp>(via);
  auto read = via.getOperand().getDefiningOp<ReadFromFifoOp>();
  if (!write || !read || !read->hasOneUse()) {
    return failure();
  }

  DataTransferOp::create(rewriter, via.getLoc(), read.getFifoSlot(),
                         write.getFifoSlot());
  rewriter.eraseOp(write);
  rewriter.eraseOp(via);
  rewriter.eraseOp(read);
  return success();
}
