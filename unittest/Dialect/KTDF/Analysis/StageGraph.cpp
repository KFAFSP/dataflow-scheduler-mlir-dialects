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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"

#include <doctest/doctest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

struct Fixture {
  explicit Fixture(MLIRContext& context);

  auto asStages(ArrayAttr attr) -> SmallVector<StageOp>;

  OwningOpRef<ModuleOp> module;
  DenseMap<StringAttr, StageOp> stages;
};

}  // namespace

Fixture::Fixture(MLIRContext& context) {
  // Construct and parse the test MLIR program.
  module = parse(&context, INPUTS_DIR "stage-graph.mlir");

  module->walk([&](StageOp stage) {
    stages[stage->getAttrOfType<StringAttr>("id")] = stage;
  });
}

auto Fixture::asStages(ArrayAttr attr) -> SmallVector<StageOp> {
  REQUIRE(attr);
  return llvm::map_to_vector(attr, [&](Attribute id) -> StageOp {
    const auto stage = stages.lookup(cast<StringAttr>(id));
    REQUIRE(stage);
    return stage;
  });
}

TEST_CASE("mlir::ktdf::StageGraph") {
  // Setup an MLIR context.
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.loadDialect<KTDFDialect>();

  Fixture fixture(context);

  for (const auto& entry : fixture.stages) {
    const auto stage_id = entry.first;
    const StageGraph::Node stage = entry.second;
    INFO("stage: ", stage_id.str());

    const auto expect_consumers = fixture.asStages(
        StageOp(stage)->getAttrOfType<ArrayAttr>("expect_consumers"));
    const auto expect_producers = fixture.asStages(
        StageOp(stage)->getAttrOfType<ArrayAttr>("expect_producers"));
    const auto expect_depends_on = fixture.asStages(
        StageOp(stage)->getAttrOfType<ArrayAttr>("expect_depends_on"));

    CHECK(unorderedEquals(llvm::to_vector(stage.getConsumers()),
                          expect_consumers));
    CHECK(unorderedEquals(llvm::to_vector(stage.getProducers()),
                          expect_producers));

    for (const auto& other : fixture.stages) {
      if (other.second == stage) {
        continue;
      }
      INFO("other: ", other.first.str());
      CHECK(stage.dependsOn(other.second) ==
            llvm::is_contained(expect_depends_on, other.second));
    }
  }
}

TEST_CASE("llvm::scc_iterator<mlir::ktdf::StageGraph::Node>") {
  // Setup an MLIR context.
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.loadDialect<KTDFDialect>();

  Fixture fixture(context);

  for (const auto& entry : fixture.stages) {
    const auto stage_id = entry.first;
    const StageGraph::Node stage = entry.second;
    INFO("entry: ", stage_id.str());

    // SCCs are emitted in reverse topological order, ending with the entry's.
    llvm::DenseSet<StageOp> visited;
    SmallVector<StageOp> last_scc;
    for (auto it = llvm::scc_begin(stage); !it.isAtEnd(); ++it) {
      const auto& scc = *it;
      REQUIRE_FALSE(scc.empty());
      INFO("scc: ", scc.front()->getAttrOfType<StringAttr>("id").str());

      CHECK(unorderedEquals(
          SmallVector<StageOp>(scc.begin(), scc.end()),
          fixture.asStages(
              scc.front()->getAttrOfType<ArrayAttr>("expect_scc"))));
      CHECK(it.hasCycle() == (scc.size() > 1));

      for (const StageGraph::Node member : scc) {
        CHECK(visited.insert(member).second);
      }
      for (const StageGraph::Node member : scc) {
        for (const auto consumer : member.getConsumers()) {
          CHECK(visited.contains(consumer));
        }
      }
      last_scc.assign(scc.begin(), scc.end());
    }

    CHECK(llvm::is_contained(last_scc, StageOp(stage)));
  }
}
