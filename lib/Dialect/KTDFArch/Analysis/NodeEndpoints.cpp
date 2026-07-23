//===-- NodeEndpoints.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/NodeEndpoints.h"

#include <mlir/Pass/AnalysisManager.h>

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// mlir::ktdf_arch::getEndpoint
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
