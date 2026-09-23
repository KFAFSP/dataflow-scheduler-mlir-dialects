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
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/PatternMatch.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir {

class DominanceInfo;
class RewritePatternSet;

}  // namespace mlir

namespace mlir::ktdf {

/// Helper class for building `ktdf.pipeline` operations.
class PipelineBuilder {
  using Token = TypedValue<TokenType>;

 public:
  using BodyBuilderFn = function_ref<void(OpBuilder&, Location)>;
  /// Indicates into which stage of a pipeline an op should be placed.
  using Placement = llvm::PointerUnion<Attribute, StageOp>;
  /// Function that decides the placement of @p op inside a pipeline.
  ///
  /// @retval nullopt   Do not put @p op in the pipeline.
  /// @retval Attribute Put @p op in the stage for the given unit/units.
  /// @retval Stage     Put @p op in this exact stage.
  using PlacementFn = function_ref<std::optional<Placement>(Operation* op)>;

  /// Creates a `ktdf.pipeline` using @p builder and obtains a builder for it.
  explicit PipelineBuilder(const OpBuilder& builder, Location loc);

  /// Gets the stage for @p units , if it exists.
  [[nodiscard]] auto getStage(ArrayAttr units) const -> StageOp {
    return units_to_stage_.lookup(units);
  }
  /// Gets the stage for @p unit_or_units , if it exists.
  [[nodiscard]] auto getStage(Attribute unit_or_units) const -> StageOp;

  /// Creates a stage.
  ///
  /// If @p units is not `nullptr`, the `applicable_units` will be set and the
  /// stage will be considered the new insertion point for that placement.
  auto createStage(ArrayAttr units = nullptr,
                   BodyBuilderFn body_builder = nullptr,
                   std::optional<Location> loc = std::nullopt) -> StageOp;

  /// Gets or creates a stage for @p units .
  auto getOrCreateStage(ArrayAttr units, BodyBuilderFn body_builder = nullptr,
                        std::optional<Location> loc = std::nullopt) -> StageOp {
    auto stage = getStage(units);
    return stage ? stage : createStage(units, body_builder, loc);
  }
  /// Gets or creates a stage for @p unit_or_units .
  auto getOrCreateStage(Attribute unit_or_units,
                        BodyBuilderFn body_builder = nullptr,
                        std::optional<Location> loc = std::nullopt) -> StageOp;

  /// Determines whether @p consumer (transitively) depends on @p producer .
  [[nodiscard]] auto hasDependency(StageOp producer, StageOp consumer) const
      -> bool {
    return dependencies_.contains({producer, consumer});
  }

  /// Adds a dependency on @p producer to @p consumer .
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(StageOp producer, StageOp consumer) -> bool;
  /// Adds a dependency between the stages of @p producer and @p consumer.
  ///
  /// @return Whether a new dependency was added.
  auto addDependency(Operation* producer, Operation* consumer) -> bool;

  /// Determines whether @p type can be forwarded between stages.
  [[nodiscard]] static auto isForwardable(Type type) -> bool;
  /// Determines whether @p type can be forwarded between stages.
  [[nodiscard]] static auto isForwardable(ShapedType type) -> bool;

  /// Attempts to forward @p value to @p consumer .
  ///
  /// If @p value is already accessible in @p consumer , it (or its last read)
  /// is returned. Otherwise, if it can be forwarded, a FIFO is created to
  /// transport the value from its producer stage to the consumer stage, and
  /// the read is returned.
  ///
  /// @retval Value   Value of @p value in @p consumer .
  /// @retval nullptr @p value can not be forwarded to @p consumer .
  [[nodiscard]] auto forwardToConsumer(Value value, StageOp consumer) -> Value;

  /// Computes the natural placement for @p op .
  ///
  /// If all users of @p op are in the same stage, this stage becomes the
  /// natural placement for @p op . Otherwise, the result is `nullopt`.
  [[nodiscard]] static auto naturalPlacement(Operation* op)
      -> std::optional<Placement>;

  /// Attempts to insert @p ops into the pipeline.
  ///
  /// Runs a work list algorithm that attempts to put @p ops and all their
  /// transitive producers into the pipeline. Evalutes @p placement for each
  /// eligible producer to determine the stage it should go to.
  void insert(ArrayRef<Operation*> ops, PlacementFn placement,
              DominanceInfo& dominance);

  /// Finalizes the pipeline, materializing the private region.
  auto finalize() -> PipelineOp;

 private:
  auto insertImpl(Operation* op, Placement placement) -> LogicalResult;

  IRRewriter rewriter_;
  PipelinePrivatizer privatizer_;
  DenseMap<ArrayAttr, StageOp> units_to_stage_;
  DenseMap<StageOp, Token> tokens_;
  DenseMap<OpResult, SmallVector<ReadFromFifoOp>> fifos_;
  DenseSet<std::pair<StageOp, StageOp>> dependencies_;
};

/// Unrolls @p via into individual hops if needed.
///
/// @pre    `rewriter` is positioned before @p via .
///
/// @return Success if the IR was modified, otherwise failure.
auto unrollVia(RewriterBase& rewriter, ViaOp via) -> LogicalResult;

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
