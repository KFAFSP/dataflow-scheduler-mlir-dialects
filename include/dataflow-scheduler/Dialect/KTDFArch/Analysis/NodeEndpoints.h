//===-- NodeEndpoints.h -----------------------------------------*- c++ -*-===//
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
//
// Every result returned from a Node is an Endpoint, which may be used by a Link
// operation to establish a link. However, not every value in the architecture
// graph is an Endpoint, since GroupOp instance can break up def-use-chains to
// enforce private resource boundaries.
//
// The Endpoint type and the getEndpoint/getNode function implement the task of
// traversing the def-use-chain until the actual declaration is found. This is
// used by the visitLinks and getLink* utilities to then look for instances of
// Link ops acting on them.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODEENDPOINTS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODEENDPOINTS_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/Pass/AnalysisManager.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"

namespace mlir::ktdf_arch {

/// Represents an endpoint of a Node.
class Endpoint : public OpResult {
 public:
  [[nodiscard]] static auto classof(Value value) -> bool {
    const auto result = dyn_cast<OpResult>(value);
    return result && classof(result);
  }
  [[nodiscard]] static auto classof(OpResult result) -> bool {
    return isa<Node>(result.getOwner());
  }

  using OpResult::OpResult;

  [[nodiscard]] auto getOwner() -> Node {
    return cast<Node>(OpResult::getOwner());
  }
};

/// Tries to obtain the Endpoint defining @p value .
///
/// Consider using the cached NodeEndpoints analysis instead.
///
/// @retval nullptr   @p value does not trace back to an Endpoint.
/// @retval Endpoint  Endpoint defining @p value .
[[nodiscard]] auto getEndpoint(Value value) -> Endpoint;

/// Tries to obtain the Node of @p NodeType defining @p value .
///
/// Consider using the cached NodeEndpoints analysis instead.
///
/// @retval nullptr   @p value is not an endpoint of a @p NodeType .
/// @retval NodeType  Node defining @p value .
template <class NodeType = Node>
[[nodiscard]] auto getNode(Value value) -> NodeType {
  if (auto endpoint = getEndpoint(value); endpoint) {
    return dyn_cast<NodeType>(endpoint.getOwner().getOperation());
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// NodeEndpoints
//===----------------------------------------------------------------------===//

/// Analysis that caches the mapping from SSA values to Endpoints.
class NodeEndpoints {
  using map_type = llvm::DenseMap<Value, Endpoint>;

 public:
  /*implicit*/ NodeEndpoints() = default;
  explicit NodeEndpoints(mlir::Operation* /*op*/) {
    // This doesn't have to be a DeviceView, so we keep it a general analysis.
  }

  static auto isInvalidated(const AnalysisManager::PreservedAnalyses& /*pa*/)
      -> bool {
    // By default, devices are considered immutable.
    return false;
  }

  /// Tries to obtain the Endpoint defining @p value .
  ///
  /// @retval nullptr   @p value does not trace back to an Endpoint.
  /// @retval Endpoint  Endpoint defining @p value .
  [[nodiscard]] auto get(Value value) -> Endpoint {
    auto [it, invalid] = endpoints_.try_emplace(value, nullptr);
    if (invalid) {
      it->second = getEndpoint(value);
    }

    return it->second;
  }
  /// @copydoc get(Value)
  [[nodiscard]] auto operator[](Value value) -> Endpoint { return get(value); }

  /// Tries to obtain the Node of @p NodeType defining @p value .
  ///
  /// @retval nullptr   @p value is not an endpoint of a @p NodeType .
  /// @retval NodeType  Node defining @p value .
  template <class NodeType = Node>
  [[nodiscard]] auto getNode(Value value) -> NodeType {
    if (auto endpoint = getEndpoint(value); endpoint) {
      return dyn_cast<NodeType>(endpoint.getOwner().getOperation());
    }
    return nullptr;
  }

 private:
  map_type endpoints_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODEENDPOINTS_H_
