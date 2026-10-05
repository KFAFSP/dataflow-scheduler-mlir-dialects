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

#include <llvm/ADT/BreadthFirstIterator.h>
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
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
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

auto PipelineBuilder::Allocator::getFifoSlotType(OpResult producer,
                                                 StageOp consumer)
    -> FifoSlotType {
  const auto value_type = cast<ShapedType>(producer.getType());
  const auto producer_unit =
      getUnitOrUnits(producer.getOwner()->getParentOfType<StageOp>());
  const auto consumer_unit = getUnitOrUnits(consumer);

  // Create the appropriate type for the slot, which flattens the elements.
  return FifoSlotType::get(producer.getContext(), producer_unit, consumer_unit,
                           value_type.getNumElements(),
                           value_type.getElementType());
}

auto PipelineBuilder::Allocator::canAllocate(OpResult producer) const -> bool {
  const auto value_type = dyn_cast<ShapedType>(producer.getType());
  return value_type && !isa<MemRefType>(value_type) &&
         value_type.hasStaticShape();
}

auto PipelineBuilder::Allocator::allocate(PipelineBuilder& builder,
                                          OpResult producer, StageOp consumer)
    -> TypedValue<FifoSlotType> {
  return cast<TypedValue<FifoSlotType>>(
      builder.getPrivateBuilder()
          .createFifo({getFifoSlotType(producer, consumer)}, {},
                      consumer->getLoc())
          .front());
}

//===----------------------------------------------------------------------===//
// PipelineBuilder::Placement
//===----------------------------------------------------------------------===//

auto PipelineBuilder::Placement::natural(Operation* op) -> Placement {
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
  return OpBuilder(&body, body.begin(), listener);
}

auto PipelineBuilder::getWriteBuilder(StageOp stage) -> OpBuilder {
  auto& body = *stage.getBody();
  if (body.empty()) {
    return OpBuilder(&body, body.end(), listener);
  }

  auto it = body.end();
  while (it != body.begin() && llvm::isa<WriteToFifoOp>(&*std::prev(it))) {
    --it;
  }
  return OpBuilder(&body, it, listener);
}

auto PipelineBuilder::build() -> PipelineOp {
  // Build the PrivateOp and erase it if it's empty.
  if (auto private_op = private_builder_.build();
      private_op.getBody()->without_terminator().empty()) {
    IRRewriter(*this).eraseOp(private_op);
  }

  return cast<PipelineOp>(getInsertionBlock()->getParentOp());
}

namespace {

/// Gets the stage @p op is in inside @p pipeline , if any.
[[nodiscard]] auto getStage(Block* pipeline, Operation* op) -> StageOp {
  for (auto stage = op->getParentOfType<StageOp>(); stage;
       stage = op->getParentOfType<StageOp>()) {
    if (stage->getBlock() == pipeline) {
      return stage;
    }
  }

  return nullptr;
}

}  // namespace

auto PipelineBuilder::getStage(Operation* op) const -> StageOp {
  return ::getStage(getInsertionBlock(), op);
}

auto PipelineBuilder::createStage(ArrayAttr units, std::optional<Location> loc,
                                  StageBuilderFn body_builder) -> StageOp {
  // Insert stages after the PrivateOp, so that producer fusion neatly orders
  // the stages in topological order.
  OpBuilder::InsertionGuard guard(*this);
  setInsertionPointAfter(private_builder_.getInsertionPoint()->getParentOp());

  auto result =
      StageOp::create(*this, loc.value_or(getLoc()), {}, {}, body_builder);
  if (units) {
    result.setApplicableUnitsAttr(units);
    units_to_stage_[units] = result;
  }
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

auto PipelineBuilder::isAvailable(const OpOperand& operand) const -> bool {
  auto* const definition = operand.get().getParentRegion();
  for (auto* region = operand.getOwner()->getParentRegion(); region != nullptr;
       region = region->getParentRegion()) {
    if (region == definition) {
      return true;
    }
    if (region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return false;
    }
  }

  auto producer = dyn_cast<OpResult>(operand.get());
  return producer && private_builder_.isPrivate(producer.getOwner());
}

auto PipelineBuilder::canSend(OpResult producer) const -> bool {
  auto stage = getStage(producer.getOwner());
  return stage && allocator_->canAllocate(producer);
}

auto PipelineBuilder::canReceive(const OpOperand& operand) const -> bool {
  auto stage = getStage(operand.getOwner());
  auto producer = dyn_cast<OpResult>(operand.get());
  return stage && producer && canSend(producer);
}

namespace {

/// Finds a `ktdf.read_from_fifo` that produces @p value in @p block .
[[nodiscard]] auto findReadIn(Value value, Block* block) -> ReadFromFifoOp {
  for (auto* const user : value.getUsers()) {
    auto write = dyn_cast<WriteToFifoOp>(user);
    if (!write || value != write.getData()) {
      continue;
    }

    for (auto& use : write.getFifoSlot().getUses()) {
      auto read = dyn_cast<ReadFromFifoOp>(use.getOwner());
      if (read && read->getBlock() != block) {
        return read;
      }
    }
  }

  return nullptr;
}

}  // namespace

auto PipelineBuilder::send(OpResult producer, StageOp consumer) -> Value {
  if (private_builder_.isPrivate(producer.getOwner())) {
    // The producer is a private op, which all stages have access to.
    return producer;
  }

  assert(canSend(producer));
  auto producer_stage = getStage(producer.getOwner());
  assert(producer_stage);

  // Lookup existing FIFO reads that produce this value.
  auto read = findReadIn(producer, consumer.getBody());
  if (!read) {
    // Allocate a new FIFO slot for this value.
    const auto slot = allocator_->allocate(*this, producer, consumer);
    assert(slot && "allocator may not fail");

    auto builder = getWriteBuilder(producer_stage);
    WriteToFifoOp::create(builder, consumer->getLoc(), producer, slot);

    // On the consumer side, create a new read from the slot.
    builder = getReadBuilder(consumer);
    read = ReadFromFifoOp::create(builder, consumer->getLoc(),
                                  producer.getType(), slot);

    // Introduce a dependency between producer and consumer via private tokens.
    addDependency(producer_stage, consumer);
  }

  return read;
}

void PipelineBuilder::receive(OpOperand& operand) {
  if (isAvailable(operand)) {
    return;
  }

  assert(canReceive(operand));
  auto consumer_stage = getStage(operand.getOwner());
  assert(consumer_stage);

  operand.set(send(cast<OpResult>(operand.get()), consumer_stage));
}

namespace {

enum class ForwardingResult : char {
  /// The result(s) can't be forwarded to the consumer(s).
  Failure = 0,
  /// Nothing needs to be forwarded.
  Skip,
  /// The result(s) can be forwarded to the consumer(s).
  Forward,
  /// The stage must be split before the result(s) can be forwarded.
  ///
  /// If forwarding the results would introduce a dependency of the producer on
  /// the consumer, the stage must be split to avoid the circular dependency.
  SplitAndForward,
};

/// Determines whether @p result can be forwarded when placed in @p stage .
[[nodiscard]] auto checkForwardingOf(
    StageOp stage, OpResult result, const PipelineBuilder::Allocator& allocator)
    -> ForwardingResult {
  auto status = ForwardingResult::Skip;

  for (auto* const user : result.getUsers()) {
    auto consumer_stage = getStage(stage->getBlock(), user);
    if (!consumer_stage) {
      // The user is not within the same pipeline.
      return ForwardingResult::Failure;
    }
    if (consumer_stage == stage) {
      // This results stays within the same stage, so no forwarding needed.
      continue;
    }

    if (StageGraph::Node(stage).dependsOn(consumer_stage)) {
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

  if (!allocator.canAllocate(result)) {
    LDBG() << "  (FAILED) result #" << result.getResultNumber()
           << " can't be forwarded";
    return ForwardingResult::Failure;
  }

  return ForwardingResult::Forward;
};

/// Determines whether @p op can be forwarded when placed in @p stage .
///
/// @pre  All users of @p result are within the pipeline.
[[nodiscard]] auto checkForwardingOf(
    StageOp stage, Operation* op, SmallVectorImpl<OpResult>& forward,
    const PipelineBuilder::Allocator& allocator) -> ForwardingResult {
  auto status = ForwardingResult::Forward;
  for (auto result : op->getResults()) {
    switch (checkForwardingOf(stage, result, allocator)) {
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

  IRRewriter rewriter(*this);

  LDBG() << "trying to insert";
  LDBG() << "    op: " << OpWithFlags(op, kSkipRegions);
  LDBG() << "  into: " << OpWithFlags(placement.stage, kSkipRegions);

  if (getStage(op)) {
    // This operation is already in the pipeline.
    LDBG() << "  (FAILED) already inside pipeline";
    return failure();
  }

  // Determine all the results that we will have to forward, checking for
  // dependency cycles in the process.
  SmallVector<OpResult> results_to_forward;
  switch (
      checkForwardingOf(placement.stage, op, results_to_forward, *allocator_)) {
    case ForwardingResult::Failure:
      return failure();
    case ForwardingResult::Forward:
    case ForwardingResult::Skip:
      break;
    case ForwardingResult::SplitAndForward: {
      // We can insert the op, but we have to break a dependency cycle. We can
      // do this by creating a new stage from the desired placement.
      auto old_stage = std::exchange(
          placement.stage, createStage(placement.stage.getApplicableUnitsAttr(),
                                       placement.stage.getLoc()));
      if (placement.erase_on_failure) {
        // This counts as a failure.
        rewriter.eraseOp(old_stage);
      }
      break;
    }
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
      receive(use);
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
  assert(placement_fn);

  IRRewriter rewriter(*this);

  // Initialize the work list in reverse order, since we're popping from the
  // back and want to keep the order (to preserve SSA property).
  llvm::SmallVector<Operation*> work_list(ops.rbegin(), ops.rend());
  while (!work_list.empty()) {
    auto* const op = work_list.pop_back_val();
    const auto placement = placement_fn(*this, op);
    if (!placement || failed(insert(op, placement))) {
      if (placement && placement.erase_on_failure) {
        rewriter.eraseOp(placement.stage);
      }
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

PipelineBuilder::PipelineBuilder(PipelineOp pipeline, Allocator* allocator,
                                 OpBuilder::Listener* listener)
    : ImplicitLocOpBuilder(pipeline.getLoc(), pipeline.getContext(), listener),
      private_builder_(pipeline, std::nullopt, listener),
      allocator_(allocator != nullptr ? allocator : &Allocator::getDefault()) {
  OpBuilder::setInsertionPointToEnd(pipeline.getBody());
}
