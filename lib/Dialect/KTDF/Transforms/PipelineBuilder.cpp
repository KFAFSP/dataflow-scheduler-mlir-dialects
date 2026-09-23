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

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

#define DEBUG_TYPE "ktdf-pipeline-builder"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

const auto kSkipRegions = OpPrintingFlags().skipRegions();

}  // namespace

PipelineBuilder::PipelineBuilder(const OpBuilder& builder, Location loc)
    : rewriter_(builder),
      privatizer_(rewriter_, PipelineOp::create(rewriter_, loc)) {
  rewriter_.setInsertionPointToStart(privatizer_.getPipeline().getBody());
}

auto PipelineBuilder::getStage(Attribute unit_or_units) const -> StageOp {
  auto units = dyn_cast_if_present<ArrayAttr>(unit_or_units);
  if (!units && unit_or_units) {
    units = ArrayAttr::get(rewriter_.getContext(), {unit_or_units});
  }

  return getStage(units);
}

auto PipelineBuilder::createStage(ArrayAttr units, BodyBuilderFn body_builder,
                                  std::optional<Location> loc) -> StageOp {
  auto result = StageOp::create(
      rewriter_, loc.value_or(privatizer_.getPipeline().getLoc()), {}, {},
      body_builder);
  if (units) {
    result.setApplicableUnitsAttr(units);
    units_to_stage_[units] = result;
  }
  rewriter_.setInsertionPoint(result);
  return result;
}

auto PipelineBuilder::getOrCreateStage(Attribute unit_or_units,
                                       BodyBuilderFn body_builder,
                                       std::optional<Location> loc) -> StageOp {
  auto units = dyn_cast_if_present<ArrayAttr>(unit_or_units);
  if (!units && unit_or_units) {
    units = rewriter_.getArrayAttr({unit_or_units});
  }

  return getOrCreateStage(units, body_builder, loc);
}

auto PipelineBuilder::addDependency(StageOp producer, StageOp consumer)
    -> bool {
  if (producer->getParentOp() != privatizer_.getPipeline() ||
      consumer->getParentOp() != privatizer_.getPipeline()) {
    return false;
  }

  auto token = tokens_.lookup(producer);
  if (!token) {
    token = privatizer_.createToken(producer->getLoc());
    rewriter_.modifyOpInPlace(producer,
                              [&]() { producer.addOutDependency(token); });
  }

  rewriter_.startOpModification(consumer);
  if (!consumer.addInDependency(token)) {
    rewriter_.cancelOpModification(consumer);
    return false;
  }
  rewriter_.finalizeOpModification(consumer);

  llvm::SmallVector<std::pair<StageOp, StageOp>> work_list{
      {producer, consumer}};
  while (!work_list.empty()) {
    const auto edge = work_list.pop_back_val();
    if (!dependencies_.insert(edge).second) {
      continue;
    }

    for (auto token : producer.getDependsIn()) {
      for (auto& use : token.getUses()) {
        if (auto transitive = dyn_cast<StageOp>(use.getOwner());
            transitive && transitive.isInDependency(use)) {
          work_list.emplace_back(transitive, consumer);
        }
      }
    }

    for (auto token : consumer.getDependsOut()) {
      for (auto& use : token.getUses()) {
        if (auto transitive = dyn_cast<StageOp>(use.getOwner());
            transitive && transitive.isOutDependency(use)) {
          work_list.emplace_back(producer, transitive);
        }
      }
    }
  }
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

namespace {

[[nodiscard]] auto getUnitOrUnits(StageOp stage) -> Attribute {
  const auto units = stage.getApplicableUnitsAttr();
  if (!units) {
    return ArrayAttr::get(stage->getContext(), {});
  }

  return units.size() == 1 ? units.getValue().front() : units;
}

void setWriteInsertionPoint(OpBuilder& builder, StageOp stage) {
  auto& body = *stage.getBody();
  if (body.empty()) {
    builder.setInsertionPointToStart(&body);
    return;
  }

  auto it = std::prev(body.end());
  while (it != body.begin() && isa<WriteToFifoOp>(&*it)) {
    --it;
  }
  builder.setInsertionPointAfter(&*it);
}

void setReadInsertionPoint(OpBuilder& builder, StageOp stage) {
  auto& body = *stage.getBody();
  auto it = body.begin();
  while (it != body.end() && isa<ReadFromFifoOp>(&*it)) {
    ++it;
  }
  builder.setInsertionPoint(&body, it);
}

}  // namespace

auto PipelineBuilder::isForwardable(Type type) -> bool {
  const auto shaped = dyn_cast<ShapedType>(type);
  return shaped && isForwardable(shaped);
}

auto PipelineBuilder::isForwardable(ShapedType type) -> bool {
  return !isa<MemRefType>(type) && type.hasStaticShape();
}

auto PipelineBuilder::forwardToConsumer(Value value, StageOp consumer)
    -> Value {
  if (value.getParentRegion()->isAncestor(consumer->getParentRegion())) {
    // The consumer already has access to the value, either because the producer
    // is within the consumer stage or because it is outside the pipeline.
    return value;
  }

  auto producer = dyn_cast<OpResult>(value);
  if (!producer) {
    // The value isn't produced by an operation, which means we can't make it
    // available to the consumer.
    return nullptr;
  }
  if (privatizer_.isPrivate(producer.getOwner())) {
    // The producer is going to be private, we don't have to do anything.
    return value;
  }
  const auto value_type = dyn_cast<ShapedType>(value.getType());
  if (!isForwardable(value_type)) {
    // We're unable to forward this value.
    // FIXME: To handle memref types, the memory allocation would have to be
    //        promoted to private memory, and the result copied.
    // FIXME: To handle dynamically-sized slots, the dimensions must be computed
    //        in front of the pipeline (or the private section).
    return nullptr;
  }
  auto producer_stage = producer.getOwner()->getParentOfType<StageOp>();
  if (!producer_stage ||
      producer_stage->getParentOp() != privatizer_.getPipeline()) {
    // We're not in charge of this producer and can't forward its value.
    return nullptr;
  }

  // Lookup existing FIFO reads that produce this value.
  auto& reads = fifos_[producer];
  for (auto read : reads) {
    if (read->getParentOp() == consumer) {
      return read;
    }
  }

  // We can't just read from an existing FIFO slot, we'll have to make a copy
  // of the value at the producer's location and plumb a new FIFO.
  const auto producer_unit = getUnitOrUnits(producer_stage);
  const auto consumer_unit = getUnitOrUnits(consumer);

  // Create the appropriate type for the slot, which flattens the elements.
  const auto slot_type = FifoSlotType::get(
      rewriter_.getContext(), producer_unit, consumer_unit,
      value_type.getNumElements(), value_type.getElementType());
  const auto slot =
      privatizer_.createFifo({slot_type}, {}, consumer->getLoc()).front();

  // On the producer side, create a new write to the slot.
  OpBuilder::InsertionGuard guard(rewriter_);
  setWriteInsertionPoint(rewriter_, producer_stage);
  WriteToFifoOp::create(rewriter_, consumer->getLoc(), producer, slot);

  // On the consumer side, create a new read from the slot.
  setReadInsertionPoint(rewriter_, consumer);
  auto read = ReadFromFifoOp::create(rewriter_, consumer->getLoc(),
                                     value.getType(), slot);
  reads.push_back(read);

  // Introduce a dependency between producer and consumer via private tokens.
  addDependency(producer_stage, consumer);
  return read;
}

namespace {

void dominanceSort(MutableArrayRef<Operation*> op, DominanceInfo& dominance) {
  llvm::stable_sort(op, [&](Operation* lhs, Operation* rhs) -> bool {
    return dominance.dominates(lhs, rhs);
  });
}

}  // namespace

auto PipelineBuilder::naturalPlacement(Operation* op)
    -> std::optional<Placement> {
  StageOp result;
  for (auto* const user : op->getUsers()) {
    auto consumer_stage = user->getParentOfType<StageOp>();
    if (!consumer_stage || (result && consumer_stage != result)) {
      return std::nullopt;
    }

    result = consumer_stage;
  }

  return result;
}

void PipelineBuilder::insert(ArrayRef<Operation*> ops, PlacementFn placement,
                             DominanceInfo& dominance) {
  // Initialize the work list in reverse order, since we're popping from the
  // back and want to keep the order (to preserve SSA property).
  llvm::SmallVector<Operation*> work_list(ops.rbegin(), ops.rend());
  while (!work_list.empty()) {
    auto* const op = work_list.pop_back_val();
    LDBG() << "trying to insert " << OpWithFlags(op, kSkipRegions);

    const auto is_in_pipeline = [&](Operation* op) -> bool {
      return privatizer_.getPipeline()->isAncestor(op) ||
             privatizer_.isPrivate(op);
    };
    if (is_in_pipeline(op)) {
      // This operation is already in the pipeline.
      LDBG() << "  (OK) already inside pipeline";
      continue;
    }
    if (!llvm::all_of(op->getUsers(), is_in_pipeline)) {
      // This operation is already in the pipeline.
      LDBG() << "  (FAIL) has users outside of pipeline";
      continue;
    }
    auto maybe_placement = placement(op);
    StageOp stage;
    if (!maybe_placement) {
      // The caller does not want this op to become part of the pipeline.
      LDBG() << "  (FAIL) filtered by caller";
      continue;
    }
    if (failed(insertImpl(op, *maybe_placement))) {
      // We were unable to insert this op, due to a failure to forward its
      // results to its consumers.
      continue;
    }

    // Consider all the inserted ops producers for insertion into the pipeline.
    const auto split = work_list.size();
    for (auto needs : op->getOperands()) {
      if (const auto wants = dyn_cast<OpResult>(needs); wants) {
        work_list.push_back(wants.getOwner());
      }
    }
    // We need to ensure that dependencies between the producers do not prevent
    // them from being inserted into the pipeline. We can prevent this from
    // happending by visiting the producers in reverse dominance order.
    dominanceSort(MutableArrayRef(work_list.data() + split, work_list.end()),
                  dominance);
  }
}

auto PipelineBuilder::finalize() -> PipelineOp {
  privatizer_.finalize();
  return privatizer_.getPipeline();
}

namespace {

enum class ForwardingResult {
  Failure = 0,
  Forward = 1,
  SplitAndForward = 2,
  Skip = 3,
};

[[nodiscard]] auto checkResult(const PipelineBuilder& builder,
                               StageOp placement, OpResult result)
    -> ForwardingResult {
  if (placement) {
    auto status = ForwardingResult::Skip;

    for (auto* const user : result.getUsers()) {
      auto consumer_stage = user->getParentOfType<StageOp>();
      assert(consumer_stage);
      if (consumer_stage == placement) {
        // This results stays within the placement stage, so doesn't need to
        // be forwarded.
        continue;
      }

      if (builder.hasDependency(consumer_stage, placement)) {
        // Forwarding this result would create a cyclic dependency.
        LDBG() << "  (WARN) detected dependency cycle between";
        LDBG() << "    " << OpWithFlags(consumer_stage, kSkipRegions);
        LDBG() << "    " << OpWithFlags(placement, kSkipRegions);

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
  }

  if (!mlir::ktdf::PipelineBuilder::isForwardable(result.getType())) {
    LDBG() << "  (FAIL) result #" << result.getResultNumber()
           << " can't be forwarded";
    return ForwardingResult::Failure;
  }

  return ForwardingResult::Forward;
};

[[nodiscard]] auto checkResults(const PipelineBuilder& builder,
                                StageOp placement, Operation* op,
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

auto PipelineBuilder::insertImpl(Operation* op, Placement placement)
    -> LogicalResult {
  // Evaluate the placement without mutating the IR. We might not be able to
  // insert the op, in which case we don't want to create a stage.
  StageOp stage;
  if (auto unit_or_units = dyn_cast_if_present<Attribute>(placement);
      unit_or_units) {
    stage = getStage(unit_or_units);
  } else if (!placement.isNull()) {
    stage = llvm::cast<StageOp>(placement);
    assert(stage->getParentOp() == privatizer_.getPipeline());
  }

  // Determine all the results that we will have to forward, checking for
  // dependency cycles in the process.
  SmallVector<OpResult> forward;
  switch (checkResults(*this, stage, op, forward)) {
    case ForwardingResult::Failure:
      return failure();
    case ForwardingResult::Forward:
      // If we don't have a stage yet, it's time to create one.
      if (!stage) {
        if (auto unit_or_units = dyn_cast_if_present<Attribute>(placement);
            unit_or_units) {
          stage = getOrCreateStage(unit_or_units);
        } else {
          stage = createStage();
        }
      }
      break;
    case ForwardingResult::SplitAndForward:
      // We can insert the op, but we have to break a dependency cycle. We can
      // do this by creating a new stage from the desired placement.
      assert(stage);
      stage =
          createStage(stage.getApplicableUnitsAttr(), nullptr, stage.getLoc());
      break;
    case ForwardingResult::Skip:
      llvm_unreachable("unexpected ForwardingResult");
  }

  // This op can safely be moved into the pipeline.
  assert(stage);
  {
    OpBuilder::InsertionGuard guard(rewriter_);
    setReadInsertionPoint(rewriter_, stage);
    op->remove();
    rewriter_.insert(op);
  }

  // Forward all results of this operation to their in-pipeline users.
  for (auto result : forward) {
    for (auto& use : result.getUses()) {
      auto stage = use.getOwner()->getParentOfType<StageOp>();
      assert(stage);
      auto forwarded = forwardToConsumer(result, stage);
      assert(forwarded);
      use.set(forwarded);
    }
  }

  LDBG() << "  (OK) inserted";
  return success();
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::unrollVia
//===----------------------------------------------------------------------===//

auto mlir::ktdf::unrollVia(RewriterBase& rewriter, ViaOp via) -> LogicalResult {
  SmallVector<Attribute> hops;
  auto source = via.collectHops(hops);
  if (hops.size() == 1) {
    return failure();
  }

  for (auto hop : ArrayRef(hops).drop_back()) {
    source = ViaOp::create(rewriter, via.getLoc(), source, hop);
  }

  auto prev = via.getOperand().getDefiningOp<ViaOp>();
  rewriter.modifyOpInPlace(via, [&]() {
    via.setOperand(source);
    via.setHopsAttr(rewriter.getArrayAttr({hops.back()}));
  });

  while (prev) {
    via = prev;
    prev = prev.getOperand().getDefiningOp<ViaOp>();

    if (via->use_empty()) {
      rewriter.eraseOp(via);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mlir::ktdf::eliminiateVia
//===----------------------------------------------------------------------===//

auto mlir::ktdf::eliminateVia(RewriterBase& rewriter, ViaOp via)
    -> LogicalResult {
  const auto users = via->getUsers();
  if (users.empty()) {
    rewriter.eraseOp(via);
    return success();
  }
  if (std::next(users.begin()) != users.end()) {
    return failure();
  }
  auto write = dyn_cast<WriteToFifoOp>(*users.begin());
  auto read = via.getOperand().getDefiningOp<ReadFromFifoOp>();
  if (!read->hasOneUse() || !write || !read) {
    return failure();
  }

  DataTransferOp::create(rewriter, via.getLoc(), read.getFifoSlot(),
                         write.getFifoSlot());
  rewriter.eraseOp(write);
  rewriter.eraseOp(via);
  rewriter.eraseOp(read);
  return success();
}
