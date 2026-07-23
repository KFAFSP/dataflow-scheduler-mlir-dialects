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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"

#include <doctest/doctest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::ResourceKinds") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "query.mlir");
  auto device = cast<DeviceOp>(module->getBody()->front());

  // Setup an AnalysisManager to mock pass execution.
  ModuleAnalysisManager module_analyses(module.get(), nullptr);
  AnalysisManager analysis_manager = module_analyses;
  auto& analysis = analysis_manager.getChildAnalysis<ResourceKinds>(device);

  llvm::DenseSet<Attribute> kinds;
  llvm::SmallVector<Resource> resources;
  device->walk([&](Resource resource) {
    resources.push_back(resource);
    if (const auto kind = resource.getKind(); kind) {
      kinds.insert(kind);
    }
  });

  Builder builder(&context);

  SUBCASE("size()") { CHECK_EQ(analysis.size(), kinds.size()); }

  SUBCASE("getInstance") {
    CHECK(analysis.getInstance(*kinds.begin()));
    CHECK(analysis.getInstance<GroupOp>(builder.getStringAttr("g0")));
    CHECK_FALSE(
        analysis.getInstance<ExecutionUnitOp>(builder.getStringAttr("g0")));
  }

  SUBCASE("getAncestors") {
    auto ancestors = analysis.getAncestors(builder.getStringAttr("e1"));
    CHECK(ancestors.erase(builder.getStringAttr("g1")));
    CHECK(ancestors.erase(builder.getStringAttr("g0")));
    CHECK(ancestors.empty());
  }

  SUBCASE("getInstances") {
    CHECK_EQ(analysis.getInstances(builder.getStringAttr("e0")).size(), 2);
    CHECK_EQ(analysis.getInstances(builder.getStringAttr("e1")).size(), 2);
  }
}
