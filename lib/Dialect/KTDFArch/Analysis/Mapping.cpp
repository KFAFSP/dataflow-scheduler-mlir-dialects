//===-- Mapping.cpp ---------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

#define DEBUG_TYPE "ktdfarch-mapping"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

const auto kSkipRegions = OpPrintingFlags().skipRegions();

}  // namespace

//===----------------------------------------------------------------------===//
// ResourceSpec
//===----------------------------------------------------------------------===//

void ResourceSpec::print(raw_ostream& os) const {
  if (isNull()) {
    os << "<<NULL>>";
    return;
  }

  if (auto resource = llvm::dyn_cast<Resource>(*this); resource) {
    if (const auto id = resource.getId(); id) {
      os << "id(" << id << ")";
      return;
    }

    os << OpWithFlags(resource, kSkipRegions);
    return;
  }

  os << "kind(" << llvm::cast<KindAttr>(*this) << ")";
}

//===----------------------------------------------------------------------===//
// Mapping
//===----------------------------------------------------------------------===//

Mapping::Mapping(const DeviceRef& device)
    : device_(device),
      by_id_(device_.getOrCreateView<ResourceIds>()),
      by_kind_(device_.getOrCreateView<ResourceKinds>()) {}

auto Mapping::resolve(MapsToAttr maps_to,
                      SmallPtrSetImpl<Resource>& resources) const
    -> LogicalResult {
  for (const auto spec : maps_to.getValue()) {
    const auto resource = lookup(spec);
    if (resource == nullptr) {
      return failure();
    }

    resources.insert(resource);
  }

  return success();
}

auto Mapping::map(ResourceSpec maps_to) -> ResourceSpecAttr {
  if (auto resource = dyn_cast<Resource>(maps_to); resource) {
    if (const auto id = by_id_.getOrAssign(resource); id) {
      return FlatSymbolRefAttr::get(id);
    }
    return nullptr;
  }

  return cast<KindAttr>(maps_to);
}

auto Mapping::map(ArrayRef<ResourceSpec> maps_to) -> MapsToAttr {
  SmallVector<ResourceSpecAttr> value;
  value.resize_for_overwrite(maps_to.size());

  for (auto [spec, attr] : llvm::zip_equal(maps_to, value)) {
    if ((attr = map(spec)) == nullptr) {
      return nullptr;
    }
  }

  return MapsToAttr::get(getContext(), value);
}

auto Mapping::map(Mappable mappable, ArrayRef<ResourceSpec> maps_to)
    -> LogicalResult {
  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "mapping " << OpWithFlags(mappable, kSkipRegions) << " to [";
    llvm::interleaveComma(maps_to, os);
    os << "]";
  })

  const auto attr = map(maps_to);
  if (!attr) {
    return failure();
  }

  const auto result = mappable.setMapsTo(attr);
  if (failed(result)) {
    LDBG() << "  >>> FAILED: op failed to setMapsTo";
  }

#ifndef NDEBUG
  return verify(mappable);
#else
  return result;
#endif
}

auto Mapping::getOrMap(Mappable mappable, ArrayRef<ResourceSpec> fallback)
    -> FailureOr<Resources> {
  Resources result;
  if (failed(resolve(mappable, result))) {
    return failure();
  }

  if (result.empty()) {
    for (const auto spec : fallback) {
      const auto resource = lookup(spec);
      if (resource == nullptr) {
        return failure();
      }

      result.insert(resource);
    }

    if (failed(map(mappable, fallback))) {
      return failure();
    }
  }

  return success(std::move(result));
}

auto Mapping::verify(Mappable mappable) const -> LogicalResult {
  const auto maps_to = mappable.getMapsTo();
  if (!maps_to) {
    return success();
  }

  SmallVector<Resource> resources;
  resources.reserve(maps_to.getValue().size());
  for (auto spec : maps_to.getValue()) {
    if (resources.emplace_back(lookup(spec)) == nullptr) {
      return mappable->emitError("invalid mapping: unresolved resource ")
             << spec;
    }
  }

  return verifyImpl(mappable, resources);
}

auto Mapping::verifyImpl(Mappable mappable, ArrayRef<Resource> resources) const
    -> LogicalResult {
  return mappable.verifyMapping(resources);
}
