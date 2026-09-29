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
#include <llvm/ADT/iterator_range.h>
#include <mlir/IR/Builders.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageDependency.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir {

class DominanceInfo;

}  // namespace mlir

namespace mlir::ktdf {

/// RAII helper that allows moving code to a PipelineOp.
///
/// Users may call `insert` on ops to attempt moving them into the pipeline.
/// Some updating of the PipelineOp is deffered until the `build()` method is
/// called or the helper is destroyed.
class PipelineBuilder : protected ImplicitLocOpBuilder {
 public:
  /// Controls the allocation of FIFO slots.
  struct Allocator {
    /// Gets the default allocator.
    [[nodiscard]] static auto getDefault() -> Allocator&;
    /// Gets the FifoSlotType between @p producer and @p consumer .
    [[nodiscard]] static auto getFifoSlotType(OpResult producer,
                                              StageOp consumer) -> FifoSlotType;

    /*implicit*/ Allocator() = default;
    /*implicit*/ Allocator(Allocator&&) = default;
    /*implicit*/ Allocator(const Allocator&) = default;
    auto operator=(Allocator&&) -> Allocator& = default;
    auto operator=(const Allocator&) -> Allocator& = default;

    virtual ~Allocator() = default;

    /// Determines whether a slot can be allocated for @p producer .
    [[nodiscard]] virtual auto canAllocate(OpResult producer) const -> bool;
    /// Allocates a slot for @p producer .
    ///
    /// This method may not fail when `canAllocate` previously returned `true`
    /// for the given @p producer .
    [[nodiscard]] virtual auto allocate(PipelineBuilder& builder,
                                        OpResult producer, StageOp consumer)
        -> FifoSlot;
  };

  /// Determines where an operation should be placed in the pipeline.
  struct Placement {
    /// Computes the natural placement for @p op .
    ///
    /// If all users of @p op are in the same stage, this stage becomes the
    /// natural placement for @p op . Otherwise, the result is `nullptr`.
    [[nodiscard]] static auto natural(Operation* op) -> Placement;

    /// Initializes a not-into-pipleine placement.
    /*implicit*/ Placement() = default;
    /// @copydoc Placement()
    /*implicit*/ Placement(std::nullptr_t) : Placement() {}
    /// Initializes a Placement into @p stage .
    ///
    /// If @p erase_on_failure is set, the PipelineBuilder will erase the
    /// stage if the placement fails.
    ///
    /// @pre  `!erase_on_failure || stage.getBody()->empty()`
    /*implicit*/ Placement(StageOp stage, bool erase_on_failure = false)
        : stage(stage), erase_on_failure(erase_on_failure) {
      assert(!erase_on_failure || stage.getBody()->empty());
    }

    /// Gets whether the operation should be placed in the pipeline.
    explicit operator bool() const { return stage != nullptr; }
    /// Gets the stage the operation should be placed in.
    /*implicit*/ operator StageOp() const { return stage; }

    /// The stage the operation shall be inserted into.
    StageOp stage;
    /// Whether the stage shall be deleted if insertion fails.
    bool erase_on_failure = false;
  };

  /// Initializes a builder on a PipelineOp created using @p builder .
  explicit PipelineBuilder(OpBuilder& builder, Location loc,
                           Allocator* allocator = nullptr)
      : PipelineBuilder(PipelineOp::create(builder, loc), allocator,
                        builder.getListener()) {}
  explicit PipelineBuilder(ImplicitLocOpBuilder& builder,
                           Allocator* allocator = nullptr)
      : PipelineBuilder(PipelineOp::create(builder), allocator,
                        builder.getListener()) {}

  /// Ensures the PipelineOp is built.
  virtual ~PipelineBuilder() { build(); }

  PipelineBuilder(PipelineBuilder&&) = delete;
  PipelineBuilder(const PipelineBuilder&) = delete;
  auto operator=(PipelineBuilder&&) = delete;
  auto operator=(const PipelineBuilder&) = delete;

  //===--------------------------------------------------------------------===//
  // Builder Interface
  //===--------------------------------------------------------------------===//

  /// Gets the underlying MLIRContext.
  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return OpBuilder::getContext();
  }

  /// Converts @p attr to a units array.
  ///
  /// If @p attr is an ArrayAttr or `nullptr`, forwards it. Otherwise, wraps
  /// @p attr in an ArrayAttr and returns that.
  [[nodiscard]] auto getUnits(Attribute attr) const -> ArrayAttr;

  /// Gets the underlying PrivateBuilder.
  [[nodiscard]] auto getPrivateBuilder() -> PrivateBuilder& {
    return private_builder_;
  }
  /// Gets a builder for contained stages.
  [[nodiscard]] auto getStageBuilder() -> ImplicitLocOpBuilder { return *this; }
  /// Gets an OpBuilder to insert reads into @p stage .
  [[nodiscard]] auto getReadBuilder(StageOp stage) -> OpBuilder;
  /// Gets an OpBuilder to insert writes into @p stage .
  [[nodiscard]] auto getWriteBuilder(StageOp stage) -> OpBuilder;

  /// Builds the PipelineOp.
  ///
  /// Applies all deferred modifications to the IR and finalizes the result.
  /// The PipelineBuilder is left in a state as if it was re-initialized on the
  /// resulting operation.
  virtual auto build() -> PipelineOp;

  //===--------------------------------------------------------------------===//
  // Stage Building
  //===--------------------------------------------------------------------===//

  using stage_iterator = Block::op_iterator<StageOp>;
  using StageBuilderFn = function_ref<void(OpBuilder&, Location)>;

  /// Gets the stage for @p units , if it exists.
  [[nodiscard]] auto getStage(ArrayAttr units) const -> StageOp {
    return units_to_stage_.lookup(units);
  }
  /// Gets the stage for @p unit_or_units , if it exists.
  [[nodiscard]] auto getStage(Attribute unit_or_units) const -> StageOp {
    return getStage(getUnits(unit_or_units));
  }
  /// Gets the stage that @p op is in, if any.
  [[nodiscard]] auto getStage(Operation* op) const -> StageOp;

  /// Gets the stages in the pipeline.
  [[nodiscard]] auto getStages() const -> iterator_range<stage_iterator> {
    return getInsertionBlock()->getOps<StageOp>();
  }

  /// Creates a new stage.
  ///
  /// If @p units is not `nullptr`, the `applicable_units` will be set and the
  /// stage will be considered the new insertion point for that placement.
  virtual auto createStage(ArrayAttr units, std::optional<Location> loc,
                           StageBuilderFn body_builder) -> StageOp;
  /// Creates a new stage.
  ///
  /// See createStage(ArrayAttr, std::optional<Location>, StageBuilderFn) for
  /// more information.
  auto createStage(ArrayAttr units = nullptr,
                   std::optional<Location> loc = std::nullopt) -> StageOp {
    return createStage(units, loc, nullptr);
  }

  /// Gets or creates a stage for @p units .
  auto getOrCreateStage(ArrayAttr units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    auto stage = getStage(units);
    return stage ? stage : createStage(units, loc);
  }
  /// Gets or creates a stage for @p unit_or_units .
  auto getOrCreateStage(Attribute unit_or_units,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    return getOrCreateStage(getUnits(unit_or_units), loc);
  }

  //===--------------------------------------------------------------------===//
  // FIFO Building
  //===--------------------------------------------------------------------===//

  /// Gets the underlying FIFO allocator.
  [[nodiscard]] auto getAllocator() const -> Allocator& { return *allocator_; }

  /// Adds a dependency on @p producer to @p consumer .
  ///
  /// @return Whether a new dependency was added.
  virtual auto addDependency(StageOp producer, StageOp consumer) -> bool;
  /// Adds a dependency between the stages of @p producer and @p consumer.
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(Operation* producer, Operation* consumer) -> bool;

  /// Determines whether @p producer is available in @p consumer .
  [[nodiscard]] auto isAvailable(OpResult producer, StageOp consumer) const
      -> bool;
  /// Determines whether @p producer can be forwarded between stages.
  [[nodiscard]] virtual auto canForward(OpResult producer) const -> bool;

  /// Forwards @p producer to @p consumer .
  ///
  /// If @p producer is already accessible in @p consumer , it (or its last
  /// read) is returned. Otherwise, if it can be forwarded, a FIFO is created to
  /// transport the value from its producer stage to the consumer stage, and
  /// the read is returned.
  ///
  /// @pre    `isAvailable(producer, consumer) || canForward(value)`
  ///
  /// @retval Value   Value of @p producer in @p consumer .
  [[nodiscard]] virtual auto forward(OpResult producer, StageOp consumer)
      -> Value;

  //===--------------------------------------------------------------------===//
  // Insertion & Placement
  //===--------------------------------------------------------------------===//

  using PlacementFn = function_ref<Placement(PipelineBuilder&, Operation*)>;

  /// Returns a Placement that will be `erased_on_failure`.
  auto tryPlacement(std::optional<Location> loc = std::nullopt) -> Placement {
    return {createStage({}, loc), true};
  }
  /// Returns a Placement for @p units that will be `erased_on_failure` if it
  /// was created.
  auto tryPlacement(ArrayAttr units, std::optional<Location> loc = std::nullopt)
      -> Placement {
    if (auto stage = getStage(units); stage) {
      return {stage, false};
    }
    return {createStage(units, loc), true};
  }
  /// Returns a Placement for @p unit_or_units that will be `erased_on_failure`
  /// if it was created.
  auto tryPlacement(Attribute unit_or_units,
                    std::optional<Location> loc = std::nullopt) -> Placement {
    return tryPlacement(getUnits(unit_or_units), loc);
  }

  /// Attempts to insert @p op into @p placement .
  virtual auto insert(Operation* op, Placement placement) -> LogicalResult;
  /// Attempts to insert @p ops into the pipeline.
  ///
  /// Runs a work list algorithm that attempts to put @p ops and all their
  /// transitive producers into the pipeline. Evalutes @p placement_fn for each
  /// eligible producer to determine the stage it should go to.
  void insert(ArrayRef<Operation*> ops, PlacementFn placement_fn,
              DominanceInfo& dominance);

 protected:
  explicit PipelineBuilder(PipelineOp pipeline, Allocator* allocator = nullptr,
                           OpBuilder::Listener* listener = nullptr);

  PrivateBuilder private_builder_;
  DenseMap<ArrayAttr, StageOp> units_to_stage_;
  DenseMap<StageOp, Token> tokens_;
  Allocator* allocator_;
  DenseMap<OpResult, SmallVector<ReadFromFifoOp>> fifos_;
  StageDependency dependencies_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PIPELINEBUILDER_H_
