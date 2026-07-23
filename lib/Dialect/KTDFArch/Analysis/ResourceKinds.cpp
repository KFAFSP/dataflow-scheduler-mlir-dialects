//===-- ResourceKinds.cpp ---------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/WalkResult.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

[[nodiscard]] auto getNearestParentKind(Resource resource) -> Attribute {
  while ((resource = resource->getParentOfType<Resource>())) {
    if (const auto kind = resource.getKind(); kind) {
      return kind;
    }
  }

  return nullptr;
}

}  // namespace

ResourceKinds::ResourceKinds(mlir::ktdf_arch::DeviceOp declaration,
                             mlir::AnalysisManager& analyses)
    : DeviceView(declaration, analyses) {
  auto& device =
      analyses
          .getAnalysis<mlir::ktdf_arch::Device, mlir::ktdf_arch::DeviceOp>();
  if (!device) {
    return;
  }

  // Visit all Resources in the device, visiting parents before children.
  device.getDefinition().walk<WalkOrder::PreOrder>(
      [&](Resource resource) -> WalkResult {
        const auto kind = resource.getKind();
        if (!kind) {
          // A kind is required for the map.
          return WalkResult::advance();
        }

        // Insert this exemplar into the map if it doesn't already exist.
        auto [it, inserted] = map_.try_emplace(kind, resource);

        // Find the nearest enclosing parent that has a kind and record it.
        if (const auto parent_kind = getNearestParentKind(resource)) {
          it->second.parent_kinds_.insert(parent_kind);
        }
        if (!inserted) {
          // We have already visited this kind, there is no need to traverse it.
          return WalkResult::skip();
        }

        return WalkResult::advance();
      });
}

void ResourceKinds::getAncestors(
    Attribute kind, llvm::SmallPtrSet<Attribute, 8>& result) const {
  const auto it = map_.find(kind);
  if (it == map_.end()) {
    return;
  }

  auto work_list = llvm::to_vector(it->second.getParentKinds());
  while (!work_list.empty()) {
    const auto next = work_list.pop_back_val();
    if (!result.insert(next).second) {
      continue;
    }

    const auto next_it = map_.find(next);
    if (next_it != map_.end()) {
      llvm::append_range(work_list, next_it->second.getParentKinds());
    }
  }
}

void ResourceKinds::getInstances(
    Attribute kind, llvm::SmallVectorImpl<Resource>& result) const {
  if (!map_.contains(kind)) {
    return;
  }

  const auto ancestors = getAncestors(kind);

  getDevice().getDefinition()->walk<WalkOrder::PreOrder>(
      [&](Resource resource) {
        const auto current_kind = resource.getKind();
        if (!current_kind) {
          return WalkResult::advance();
        }

        if (current_kind == kind) {
          result.push_back(resource);
          // Resource can not appear nested within itself.
          return WalkResult::skip();
        }

        // We know all ancestors with kinds, so we can skip all subgraphs that
        // aren't in that set.
        return ancestors.contains(current_kind) ? WalkResult::advance()
                                                : WalkResult::skip();
      });
}
