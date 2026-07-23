//===-- Links.cpp -----------------------------------------------*- c++ -*-===//
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

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// Endpoints
//===----------------------------------------------------------------------===//

auto mlir::ktdf_arch::getEndpoint(Value value) -> Endpoint {
  while (true) {
    // If the value is an argument to a GroupOp body, it is a shared memory, and
    // we need to find the endpoint that defines the capture operand.
    if (auto arg = dyn_cast<BlockArgument>(value); arg) {
      auto group = dyn_cast<GroupOp>(arg.getOwner()->getParentOp());
      if (!group) {
        break;
      }

      value = group->getOperand(arg.getArgNumber());
      continue;
    }

    // If the value is a result of a GroupOp, it is a shared execution unit, and
    // we need to find the endpoint that defines the yield operand.
    if (auto result = dyn_cast<OpResult>(value); result) {
      if (auto group = dyn_cast<GroupOp>(result.getOwner()); group) {
        value = group.getBody()->getTerminator()->getOperand(
            result.getResultNumber());
        continue;
      }

      if (isa<Node>(result.getOwner())) {
        return cast<Endpoint>(result);
      }
    }

    break;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Links
//===----------------------------------------------------------------------===//

namespace {

auto visitLinksImpl(Value value,
                    function_ref<bool(Link, LinkDirection)> callback) -> bool {
  // Visit all uses of the value to find connected links.
  for (auto& use : value.getUses()) {
    // If the value is used by a GroupOp, it is a shared memory, and we need to
    // descend into the GroupOp to find more links.
    if (auto group = dyn_cast<GroupOp>(use.getOwner()); group) {
      if (!visitLinksImpl(group.getBody()->getArgument(use.getOperandNumber()),
                          callback)) {
        return false;
      }
      continue;
    }

    // If the value is used by a YieldOp, it is a shared execution unit, and we
    // need to ascend from the GroupOp to find more links.
    if (auto yield = dyn_cast<YieldOp>(use.getOwner()); yield) {
      if (!visitLinksImpl(
              yield.getParentOp()->getResult(use.getOperandNumber()),
              callback)) {
        return false;
      }
      continue;
    }

    // Visit the use if it is a Link.
    auto link = dyn_cast<Link>(use.getOwner());
    if (!link) {
      continue;
    }

    // Determine how the Link uses the value. Links don't necessarily have to
    // connect with all value they use.
    LinkDirection direction{};
    if (llvm::is_contained(link.getSources(), value)) {
      direction |= LinkDirection::Outgoing;
    }
    if (llvm::is_contained(link.getTargets(), value)) {
      direction |= LinkDirection::Incoming;
    }
    if (direction == LinkDirection{}) {
      continue;
    }

    // Invoke the callback, determining whether visiting should be aborted.
    if (!callback(link, direction)) {
      return false;
    }
  }

  // Visited all links without aborting.
  return true;
}

}  // namespace

auto mlir::ktdf_arch::detail::visitLinks(
    Endpoint endpoint, function_ref<bool(Link, LinkDirection)> callback)
    -> bool {
  return visitLinksImpl(endpoint, callback);
}
