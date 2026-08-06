//===-- ResourceIds.cpp -----------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/WalkResult.h>

#include <string>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

ResourceIds::ResourceIds(const Device& device) : DeviceView(device) {
  // Visit all resources in the device to populate the map.
  getDevice().getBodyRegion().walk([&](Resource resource) {
    if (const auto id_attr = resource.getIdAttr(); id_attr) {
      auto [it, inserted] = map_.try_emplace(id_attr, resource);
      assert(inserted);
    }
  });
}

auto ResourceIds::assign(Resource resource, StringAttr id) -> bool {
  // We can only assign identifiers to resources owned by our device.
  if (!getDevice() || !getDevice().getDefinition()->isAncestor(resource)) {
    return false;
  }

  if (!id) {
    // Removing the identifier always succeeds.
    if (const auto id_attr = resource.getIdAttr(); id_attr) {
      map_.erase(id_attr);
      resource.removeIdAttr();
    }
    return true;
  }

  // Can't override an identifier that is already in use by someone else.
  if (const auto existing = map_.lookup(id); existing) {
    return existing == resource;
  }

  // Remove the old mapping and attach the new identifier.
  if (const auto id_attr = resource.getIdAttr(); id_attr) {
    map_.erase(id_attr);
  }
  resource.setIdAttr(id);
  map_[id] = resource;
  return true;
}

auto ResourceIds::assign(Resource resource, StringRef prefix) -> StringAttr {
  // Come up with a prefix for the name.
  llvm::SmallString<32> id(prefix);
  const auto prefix_len = id.size();

  // Make the id unique by counting up an index (but don't include 0).
  StringAttr id_attr;
  std::size_t index = 0;
  while (true) {
    id_attr = StringAttr::get(resource->getContext(), id);
    const auto existing = map_.lookup(id_attr);
    if (!existing || existing == resource) {
      break;
    }

    id.resize(prefix_len);
    id += '_';
    id += std::to_string(++index);
  }

  return assign(resource, id_attr) ? id_attr : nullptr;
}

auto ResourceIds::getOrAssign(Resource resource,
                              std::optional<StringRef> prefix) -> StringAttr {
  // Try to get the assigned identifier.
  if (const auto id_attr = resource.getIdAttr(); id_attr) {
    return id_attr;
  }

  if (!prefix) {
    // Come up with a default prefix.
    if (const auto str_kind =
            dyn_cast_if_present<StringAttr>(resource.getKind());
        str_kind) {
      // A string-based kind is ideal for the name.
      prefix = str_kind;
    } else {
      // Otherwise, we use the mnemonic of the operation.
      const auto qualified_name = resource->getName().getStringRef();
      prefix = qualified_name.drop_front(qualified_name.find(".") + 1);
    }
  }

  return assign(resource, *prefix);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::ktdf_arch::ResourceIds);
