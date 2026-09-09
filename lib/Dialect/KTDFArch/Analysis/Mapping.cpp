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

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"
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

  os << "kind(" << llvm::cast<Attribute>(*this) << ")";
}

//===----------------------------------------------------------------------===//
// Mapping
//===----------------------------------------------------------------------===//

auto Mapping::isMappable(Operation* op) -> bool {
  return !isa<KTDFArchDialect>(op->getDialect()) &&
         op->getParentOfType<DeviceOp>() == nullptr;
}

auto Mapping::get(Operation* op) -> std::pair<Operation*, MapsToAttr> {
  for (; op != nullptr; op = op->getParentOp()) {
    if (auto result = getProperty<MapsToAttr>(op); result) {
      if (result.isUnmapped()) {
        // The UnitAttr is interpreted as "no mapping", so that we can un-map
        // an operation nested beneath another if needed.
        break;
      }
      return {op, result};
    }
  }

  return {nullptr, nullptr};
}

Mapping::Mapping(DeviceOp device, AnalysisManager analyses)
    : device_(device, analyses),
      by_id_(device_.getOrCreateView<ResourceIds>()),
      by_kind_(device_.getOrCreateView<ResourceKinds>()) {}

auto Mapping::lookup(MapsToAttr maps_to) const -> Resource {
  if (auto id_ref = dyn_cast<FlatSymbolRefAttr>(maps_to); id_ref) {
    return by_id_[id_ref.getAttr()];
  }

  return by_kind_[maps_to].getExemplar();
}

void Mapping::map(Operation* mappable, ResourceSpec maps_to) {
  if (maps_to.isNull()) {
    LDBG() << "unmapping " << OpWithFlags(mappable, kSkipRegions);
    unset(mappable);
    return;
  }

  LDBG() << "mapping " << OpWithFlags(mappable, kSkipRegions) << " to "
         << maps_to;
  set(mappable,
      llvm::TypeSwitch<ResourceSpec, MapsToAttr>(maps_to)
          .Case([&](Resource resource) {
            assert(getDevice() &&
                   getDevice().getDefinition()->isAncestor(resource));
            return MapsToAttr::id(by_id_.getOrAssign(resource));
          })
          .Case([](Attribute kind) { return MapsToAttr::kind(kind); }));
}
