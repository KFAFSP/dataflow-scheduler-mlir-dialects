//===----------------------------------------------------------------------===//
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

#include <doctest/doctest.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

TEST_CASE("mlir::ktdf::PipelineBuilder") {
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.loadDialect<KTDFDialect>();
  OpBuilder builder(&context);
  auto module =
      OwningOpRef<ModuleOp>(ModuleOp::create(builder.getUnknownLoc()));
  builder.setInsertionPointToEnd(module->getBody());

  // Building erases the empty PrivateOp, and the builders build again when
  // they are destroyed.
  SUBCASE("empty pipeline, destroyed") {
    {
      PipelineBuilder pipeline_builder(builder, builder.getUnknownLoc());
    }
    auto pipelines = module->getOps<PipelineOp>();
    REQUIRE_FALSE(pipelines.empty());
    CHECK_FALSE((*pipelines.begin()).getPrivateOp());
  }

  SUBCASE("empty pipeline, built then destroyed") {
    PipelineOp pipeline;
    {
      PipelineBuilder pipeline_builder(builder, builder.getUnknownLoc());
      pipeline = pipeline_builder.build();
      CHECK_FALSE(pipeline.getPrivateOp());
      CHECK(pipeline_builder.build() == pipeline);
    }
    CHECK_FALSE(pipeline.getPrivateOp());
  }

  // The producer can only go in once both users are in the pipeline, so its
  // first placement fails and is erased.
  SUBCASE("two results, users inserted one at a time") {
    const auto loc = builder.getUnknownLoc();
    const auto type = RankedTensorType::get({4}, builder.getF32Type());

    OperationState producer_state(loc, "test.producer");
    producer_state.addTypes({type, type});
    auto* producer = builder.create(producer_state);
    const auto create_user = [&](Value operand) {
      OperationState state(loc, "test.user");
      state.addOperands(operand);
      return builder.create(state);
    };
    auto* user_0 = create_user(producer->getResult(0));
    auto* user_1 = create_user(producer->getResult(1));

    DominanceInfo dominance;
    PipelineBuilder pipeline_builder(builder, loc);
    const auto users_units = builder.getStringAttr("users");
    const auto producer_units = builder.getStringAttr("producer");
    const auto place = [&](PipelineBuilder& pipeline_builder,
                           Operation* op) -> PipelineBuilder::Placement {
      if (op == producer) {
        return pipeline_builder.tryPlacement(producer_units);
      }
      return {pipeline_builder.tryPlacement(users_units).stage, false};
    };

    pipeline_builder.insert({user_0}, place, dominance);
    CHECK_FALSE(pipeline_builder.getStage(producer));
    pipeline_builder.insert({user_1}, place, dominance);
    CHECK(pipeline_builder.getStage(producer));
    CHECK(pipeline_builder.getStage(producer) ==
          pipeline_builder.getStage(producer_units));
  }
}
