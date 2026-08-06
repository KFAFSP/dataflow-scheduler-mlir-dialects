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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceIds.h"

#include <doctest/doctest.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/WalkResult.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::ResourceIds") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "resource-ids.mlir");
  auto device = cast<DeviceOp>(module->getBody()->front());
  OpBuilder builder(&context);
  auto foreign_resource =
      ExecutionUnitOp::create(builder, builder.getUnknownLoc());

  // Setup an AnalysisManager to mock pass execution.
  ModuleAnalysisManager module_analyses(module.get(), nullptr);
  AnalysisManager analysis_manager = module_analyses;
  auto& devices = analysis_manager.getAnalysis<DeviceManager>();
  auto& resources = devices.getOrCreateView<ResourceIds>(device);

  SUBCASE("lookup(StringRef)") {
    CHECK_FALSE(resources["exec_unit"]);
    CHECK(resources["exec_unit_1"]);
  }

  SUBCASE("assign(Resource, StringAttr{}) on foreign resource") {
    CHECK_FALSE(resources.assign(foreign_resource, StringAttr{}));
  }

  SUBCASE("assign(Resource, StringRef) same name") {
    auto resource = resources["exec_unit_1"];

    CHECK(resources.assign(resource, "exec_unit_1"));
    CHECK_EQ(resource.getId(), "exec_unit_1");
    CHECK_EQ(resources["exec_unit_1"], resource);
  }

  SUBCASE("assign(Resource, StringRef) conflict") {
    auto resource = resources["exec_unit_1"];
    auto group = *device.getBody()->getOps<GroupOp>().begin();
    const auto group_id = resources.getOrAssign(group);

    CHECK_FALSE(resources.assign(resource, group_id));
    CHECK_EQ(resource.getId(), "exec_unit_1");
    CHECK_EQ(resources["exec_unit_1"], resource);
    CHECK_EQ(group.getIdAttr(), group_id);
    CHECK_EQ(resources[group_id], group);
  }

  SUBCASE("assign(Resource, StringRef) rename") {
    auto resource = resources["exec_unit_1"];

    CHECK(resources.assign(resource, "exec_unit_2"));
    CHECK_EQ(resource.getId(), "exec_unit_2");
    CHECK_FALSE(resources["exec_unit_1"]);
    CHECK_EQ(resources["exec_unit_2"], resource);
  }

  SUBCASE("getOrAssign(Resource) on foreign resource") {
    CHECK_FALSE(resources.getOrAssign(foreign_resource));
  }

  SUBCASE("getOrAssign(Resource)") {
    device->walk([&](Resource resource) {
      auto id = resources.getOrAssign(resource);
      CHECK_EQ(id, resource.getIdAttr());
      CHECK_EQ(id, resource->getAttr("expected_id"));
    });
  }

  SUBCASE("getOrAssign(Resource, StringRef)") {
    const auto prefix = StringRef("test");
    std::string expect(prefix);
    const auto prefix_len = expect.size();

    auto index = 0U;
    device->walk([&](Resource resource) {
      if (resource.getId()) {
        return;
      }

      if (index > 0) {
        expect.resize(prefix_len);
        expect += '_';
        expect += std::to_string(index);
      }
      ++index;

      auto id = resources.getOrAssign(resource, prefix);
      CHECK_EQ(id, resource.getIdAttr());
      CHECK_EQ(id, expect);
    });
  }
}