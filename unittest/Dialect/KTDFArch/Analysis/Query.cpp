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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Query.h"

#include <doctest/doctest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::Query") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "query.mlir");
  auto device = cast<DeviceOp>(module->getBody()->front());
  llvm::SmallVector<Resource> resources;
  device->walk([&](Resource resource) { resources.push_back(resource); });

  SUBCASE("where") {
    auto result =
        Query(resources)
            .where([](Resource r) -> bool { return r.getId() == "special"; })
            .singular();
    REQUIRE(result);
    CHECK(result.getId() == "special");
  }

  SUBCASE("unique") {
    const auto expected = resources.size();
    resources.push_back(resources.front());
    CHECK(Query(resources).unique().size() == expected);
  }

  SUBCASE("ofType") {
    const auto expected = llvm::count_if(
        resources, [](Resource r) -> bool { return isa<ExecutionUnitOp>(r); });
    auto exec_units = Query(resources).ofType<ExecutionUnitOp>();
    CHECK(exec_units.size() == expected);
  }

  SUBCASE("withAttribute(StringRef)") {
    CHECK(Query(resources).withAttribute("test.attr").singular());
  }

  SUBCASE("withAttribute(NamedAttribute)") {
    Builder builder(&context);
    CHECK(Query(resources)
              .withAttribute(
                  NamedAttribute("test.attr", builder.getI64IntegerAttr(4)))
              .singular());
    CHECK_FALSE(Query(resources).withAttribute(
        NamedAttribute("test.attr", builder.getI64IntegerAttr(3))));
  }

  SUBCASE("withProperty()") {
    CHECK(Query(resources).withProperty<BandwidthAttr>().singular());
  }

  SUBCASE("withProperty(PropertyAttr)") {
    CHECK(Query(resources)
              .withProperty(BandwidthAttr::get(&context, 128))
              .singular());
    CHECK_FALSE(
        Query(resources).withProperty(BandwidthAttr::get(&context, 127)));
  }

  SUBCASE("withFeature()") {
    const auto expect = llvm::count_if(resources, [](Resource r) -> bool {
      return r.getFeature<feature::Compute>() != nullptr;
    });
    CHECK_EQ(Query(resources).withFeature<feature::Compute>().size(), expect);
  }

  SUBCASE("withFeature(FeatureAttr)") {
    Builder builder(&context);
    const auto f1 =
        cast<feature::SIMD>(builder.getDictionaryAttr({NamedAttribute(
            "lanes", feature::SIMD::LanesAttr::get(
                         &context, {{TypeAttr::get(builder.getF16Type()),
                                     I64Attr::get(&context, 4)}}))}));
    const auto f2 =
        cast<feature::SIMD>(builder.getDictionaryAttr({NamedAttribute(
            "lanes", feature::SIMD::LanesAttr::get(
                         &context, {{TypeAttr::get(builder.getF16Type()),
                                     I64Attr::get(&context, 8)}}))}));

    CHECK_EQ(Query(resources).withFeature(f1).size(), 2);
    CHECK(Query(resources).withFeature(f2).singular());
  }
}
