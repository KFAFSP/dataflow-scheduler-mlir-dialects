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
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

template <class T>
[[nodiscard]] auto unorderedEquals(SmallVector<T> lhs, SmallVector<T> rhs)
    -> bool {
  llvm::sort(lhs);
  llvm::sort(rhs);
  return lhs == rhs;
}

}  // namespace

TEST_CASE("mlir::ktdf_arch::getLink*") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "links.mlir");
  auto device = *module->getOps<DeviceOp>().begin();

  DenseMap<StringAttr, Resource> nodes;
  DenseMap<StringAttr, Link> links;
  device.walk([&](Resource node) {
    if (auto link = dyn_cast<Link>(node.getOperation()); link) {
      if (const auto id_attr = link.getIdAttr(); id_attr) {
        links[id_attr] = link;
      }
      return;
    }
    if (isa<GroupOp>(node)) {
      return;
    }
    if (const auto id_attr = node.getIdAttr(); id_attr) {
      nodes[id_attr] = node;
    }
  });

  const auto as_links = [&](ArrayAttr attr_or_null) -> SmallVector<Link> {
    if (!attr_or_null) {
      return {};
    }

    return llvm::map_to_vector(attr_or_null, [&](Attribute attr) -> Link {
      const auto link_ref = cast<FlatSymbolRefAttr>(attr);
      return links[link_ref.getAttr()];
    });
  };

  for (const auto& pair : nodes) {
    const auto node_id = pair.first;
    const auto node = pair.second;
    INFO("node: ", node_id.str());

    auto expect_incoming =
        as_links(node->getAttrOfType<ArrayAttr>("expect_incoming"));
    auto expect_outgoing =
        as_links(node->getAttrOfType<ArrayAttr>("expect_outgoing"));
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
        const auto expect = as_links(refs);
        const auto got = getLinks(node, nodes[to.getAttr()]);
        CHECK(unorderedEquals(got, expect));
        if (got.size() == 1) {
          CHECK(getLink(node, nodes[to.getAttr()]) == got.front());
        }
      }
    }
  }
}
