//===-- PipelineBuilder.h ---------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_

#include <llvm/ADT/PointerUnion.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageDependency.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

namespace mlir {

class DominanceInfo;
class RewritePatternSet;

}  // namespace mlir

namespace mlir::ktdf {

/// Helper class for building `ktdf.pipeline` operations.
class PipelineBuilder : public PipelinePrivatizer {
 public:
  using BodyBuilderFn = function_ref<void(OpBuilder&, Location)>;

  /// Controls the allocation of FIFO slots.
  struct Allocator {
    /// Gets the default allocator.
    [[nodiscard]] static auto getDefault() -> Allocator&;

    virtual ~Allocator() = default;

    /// Determines whether a slot can be allocated for @p producer .
    [[nodiscard]] virtual auto canAllocate(OpResult producer) const -> bool;
    /// Allocates a slot for @p producer .
    ///
    /// This method may not fail when `canAllocate` previously returned `true`
    /// for the given @p producer .
    [[nodiscard]] virtual auto allocate(PipelineBuilder& builder,
                                        OpResult producer, StageOp consumer)
        -> TypedValue<FifoSlotType>;
  };

  /// Determines where an operation should be placed in the pipeline.
  struct Placement {
    /// Initializes a not-into-pipleine placement.
    /*implicit*/ Placement() = default;
    /// @copydoc Placement()
    /*implicit*/ Placement(std::nullptr_t) : Placement() {}
    /// Initializes a Placement into @p stage .
    ///
    /// If @p erase_on_failure is set, the PipelineBuilder will erase the
    /// stage if the placement fails.
    /*implicit*/ Placement(StageOp stage, bool erase_on_failure = false)
        : stage(stage), erase_on_failure(erase_on_failure) {}

    /// Gets whether the operation should be placed in the pipeline.
    explicit operator bool() const { return stage != nullptr; }
    /// Gets the stage the operation should be placed in.
    /*implicit*/ operator StageOp() const { return stage; }

    StageOp stage;
    bool erase_on_failure;
  };

  using PlacementFn = function_ref<Placement(PipelineBuilder&, Operation*)>;

  /// Creates a `ktdf.pipeline` using @p builder and obtains a builder for it.
  explicit PipelineBuilder(OpBuilder& builder, Location loc,
                           Allocator* allocator = nullptr)
      : PipelineBuilder(PipelineOp::create(builder, loc), allocator,
                        builder.getListener()) {}

  ~PipelineBuilder() override { finalize(); }

  /// Converts @p attr to a units array.
  ///
  /// If @p attr is an ArrayAttr or `nullptr`, forwards it. Otherwise, wraps
  /// @p attr in an ArrayAttr and returns that.
  [[nodiscard]] auto toUnits(Attribute attr) const -> ArrayAttr;

  /// Gets the stage for @p units , if it exists.
  [[nodiscard]] auto getStage(ArrayAttr units) const -> StageOp {
    return units_to_stage_.lookup(units);
  }
  /// Gets the stage for @p unit_or_units , if it exists.
  [[nodiscard]] auto getStage(Attribute unit_or_units) const -> StageOp {
    return getStage(toUnits(unit_or_units));
  }

  /// Creates a stage.
  ///
  /// If @p units is not `nullptr`, the `applicable_units` will be set and the
  /// stage will be considered the new insertion point for that placement.
  auto createStage(ArrayAttr units = nullptr,
                   BodyBuilderFn body_builder = nullptr,
                   std::optional<Location> loc = std::nullopt) -> StageOp;

  /// Gets or creates a stage for @p units .
  auto getOrCreateStage(ArrayAttr units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    auto stage = getStage(units);
    return stage ? stage : createStage(units, {}, loc);
  }
  /// Gets or creates a stage for @p unit_or_units .
  auto getOrCreateStage(Attribute unit_or_units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    return getOrCreateStage(toUnits(unit_or_units), loc);
  }

  /// Returns a Placement that will be `erased_on_failure`.
  auto tryPlacement(std::optional<Location> loc = std::nullopt) -> Placement {
    return {createStage({}, {}, loc), true};
  }
  /// Returns a Placement for @p units that will be `erased_on_failure` if it
  /// was created.
  auto tryPlacement(ArrayAttr units, std::optional<Location> loc = std::nullopt)
      -> Placement {
    if (auto stage = getStage(units); stage) {
      return {stage, false};
    }
    return {createStage(units, {}, loc), true};
  }
  /// Returns a Placement for @p unit_or_units that will be `erased_on_failure`
  /// if it was created.
  auto tryPlacement(Attribute unit_or_units,
                    std::optional<Location> loc = std::nullopt) -> Placement {
    return tryPlacement(toUnits(unit_or_units), loc);
  }

  /// Determines whether @p consumer (transitively) depends on @p producer .
  [[nodiscard]] auto hasDependency(StageOp producer, StageOp consumer) -> bool {
    return dependencies_.contains(consumer, producer, true);
  }

  /// Adds a dependency on @p producer to @p consumer .
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(StageOp producer, StageOp consumer) -> bool;
  /// Adds a dependency between the stages of @p producer and @p consumer.
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(Operation* producer, Operation* consumer) -> bool;

  /// Get the underlying FIFO allocator.
  [[nodiscard]] auto getAllocator() const -> Allocator& { return *allocator_; }

  /// Determines whether @p producer is available in @p consumer .
  [[nodiscard]] auto isAvailable(OpResult producer, StageOp consumer) const
      -> bool;
  /// Determines whether @p producer can be forwarded between stages.
  [[nodiscard]] auto canForward(OpResult producer) const -> bool;

  /// Forwards @p producer to @p consumer .
  ///
  /// If @p producer is already accessible in @p consumer , it (or its last
  /// read) is returned. Otherwise, if it can be forwarded, a FIFO is created to
  /// transport the value from its producer stage to the consumer stage, and
  /// the read is returned.
  ///
  /// @pre    `isAvailable(producer, consumer) || isForwardable(value)`
  ///
  /// @retval Value   Value of @p producer in @p consumer .
  [[nodiscard]] auto forward(OpResult producer, StageOp consumer) -> Value;

  /// Computes the natural placement for @p op .
  ///
  /// If all users of @p op are in the same stage, this stage becomes the
  /// natural placement for @p op . Otherwise, the result is `nullopt`.
  [[nodiscard]] static auto naturalPlacement(Operation* op) -> Placement;

  auto insert(Operation* op, StageOp stage) -> LogicalResult;
  auto insert(Operation* op, Placement placement) -> LogicalResult;
  /// Attempts to insert @p ops into the pipeline.
  ///
  /// Runs a work list algorithm that attempts to put @p ops and all their
  /// transitive producers into the pipeline. Evalutes @p placement_fn for each
  /// eligible producer to determine the stage it should go to.
  void insert(ArrayRef<Operation*> ops, PlacementFn placement_fn,
              DominanceInfo& dominance);

  /// Finalizes the outstanding modifications to the pipeline.
  ///
  /// If there are no modifications to perform, does nothing. After finalizing,
  /// the PipelineBuilder will be ready again to queue more modifications to
  /// the same pipeline.
  auto finalize() -> PipelineOp override;

 protected:
  explicit PipelineBuilder(PipelineOp pipeline, Allocator* allocator = nullptr,
                           OpBuilder::Listener* listener = nullptr);

  void setInsertPointToWrite(StageOp stage);
  void setInsertPointToRead(StageOp stage);

  virtual void erase(StageOp stage);

  DenseMap<ArrayAttr, StageOp> units_to_stage_;
  DenseMap<StageOp, Token> tokens_;
  Allocator* allocator_;
  DenseMap<OpResult, SmallVector<ReadFromFifoOp>> fifos_;
  StageDependency dependencies_;
};

/// Eliminates @p via if possible.
///
/// If @p via has no users, it is erased. If @p via is the single user of a
/// ReadFromFifoOp, and has a single use in a WriteToFifoOp, it is replaced with
/// a DataTransferOp instead, erasing all three ops.
///
/// @pre  `rewriter` is positioned before @p via .
///
/// @return Success if the IR was modified, otherwise failure.
auto eliminateVia(RewriterBase& rewriter, ViaOp via) -> LogicalResult;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_
