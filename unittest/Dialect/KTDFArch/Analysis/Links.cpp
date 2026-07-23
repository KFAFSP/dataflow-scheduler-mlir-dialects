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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Links.h"

#include <doctest/doctest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

struct Fixture {
  explicit Fixture(MLIRContext& context);

  auto asLinks(ArrayAttr attr_or_null) -> SmallVector<Link>;

  OwningOpRef<ModuleOp> module;
  DeviceOp device;
  DenseMap<StringAttr, Node> nodes;
  DenseMap<StringAttr, Link> links;
};

}  // namespace

Fixture::Fixture(MLIRContext& context) {
  // Construct and parse the test MLIR program.
  module = parse(&context, INPUTS_DIR "links.mlir");
  device = *module->getOps<DeviceOp>().begin();

  device.walk([&](Resource resource) {
    if (auto link = dyn_cast<Link>(resource.getOperation()); link) {
      if (const auto id_attr = link.getIdAttr(); id_attr) {
        links[id_attr] = link;
      }
      return;
    }
    if (auto node = dyn_cast<Node>(resource.getOperation()); node) {
      if (const auto id_attr = node.getIdAttr(); id_attr) {
        nodes[id_attr] = node;
      }
    }
  });
}

auto Fixture::asLinks(ArrayAttr attr_or_null) -> SmallVector<Link> {
  if (!attr_or_null) {
    return {};
  }

  return llvm::map_to_vector(attr_or_null, [&](Attribute attr) -> Link {
    const auto link_ref = cast<FlatSymbolRefAttr>(attr);
    return links[link_ref.getAttr()];
  });
};

TEST_CASE("mlir::ktdf_arch::getLink*") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  Fixture fixture(context);

  for (const auto& pair : fixture.nodes) {
    const auto node_id = pair.first;
    const auto node = pair.second;
    INFO("node: ", node_id.str());

    auto expect_incoming =
        fixture.asLinks(node->getAttrOfType<ArrayAttr>("expect_incoming"));
    auto expect_outgoing =
        fixture.asLinks(node->getAttrOfType<ArrayAttr>("expect_outgoing"));
    auto expect =
        llvm::to_vector(llvm::concat<Link>(expect_incoming, expect_outgoing));

    auto all_links = getLinks(node);
    CHECK(unorderedEquals(all_links, expect));
    if (all_links.size() == 1) {
      CHECK(getLink(node) == all_links.front());
    }

    auto incoming_links = getLinks(LinkDirection::Incoming, node);
    CHECK(unorderedEquals(incoming_links, expect_incoming));
    if (incoming_links.size() == 1) {
      CHECK(getLink(LinkDirection::Incoming, node) == incoming_links.front());
    }

    auto outgoing_links = getLinks(LinkDirection::Outgoing, node);
    CHECK(unorderedEquals(outgoing_links, expect_outgoing));
    if (outgoing_links.size() == 1) {
      CHECK(getLink(LinkDirection::Outgoing, node) == outgoing_links.front());
    }

    if (const auto expect_to =
            node->getAttrOfType<TypedMapAttr<FlatSymbolRefAttr, ArrayAttr>>(
                "expect_to");
        expect_to) {
      for (const auto& pair2 : expect_to) {
        const auto to = pair2.first;
        const auto refs = pair2.second;
        INFO("to: ", to.getAttr().str());
        const auto expect = fixture.asLinks(refs);
        const auto got = getLinks(node, fixture.nodes[to.getAttr()]);
        CHECK(unorderedEquals(got, expect));
        if (got.size() == 1) {
          CHECK(getLink(node, fixture.nodes[to.getAttr()]) == got.front());
        }
      }
    }
  }
}
