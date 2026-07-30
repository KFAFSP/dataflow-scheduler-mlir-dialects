//===-- KTDFArchInterfaces.cpp ----------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

auto mlir::ktdf_arch::verifyResource(Operation* op) -> LogicalResult {
  // All resources must be nested under other resources (or devices).
  auto* const parent = op->getParentOp();
  if (!isa<DeviceOp, Resource>(parent)) {
    return op->emitOpError("expects parent op to be '")
           << DeviceOp::getOperationName() << "' or another resource";
  }

  return success();
}

auto mlir::ktdf_arch::verifyLink(Operation* op) -> LogicalResult {
  // All links must be nested under a resource (or device).
  auto* const parent = op->getParentOp();
  if (!isa<DeviceOp, Resource>(parent)) {
    return op->emitOpError("expects parent op to be '")
           << DeviceOp::getOperationName() << "' or a resource";
  }

  // All links must have at least one source and one target.
  auto iface = cast<Link>(op);
  if (iface.getSources().empty()) {
    return op->emitOpError("requires one source endpoint");
  }
  if (iface.getTargets().empty()) {
    return op->emitOpError("requires one target endpoint");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mlir::ktdf_arch::getFeature
//===----------------------------------------------------------------------===//

auto mlir::ktdf_arch::getFeature(Operation* op, const Feature& required)
    -> std::optional<Feature> {
  // Find a provided feature of the same name and perform a test.
  auto maybe_provided = getFeature(op, required.getName());
  if (maybe_provided && required.matchedBy(maybe_provided->getValue())) {
    return *maybe_provided;
  }

  return std::nullopt;
}

auto mlir::ktdf_arch::getFeature(Operation* op, StringRef name)
    -> std::optional<Feature> {
  const auto features = getProperty<FeaturesAttr>(op);
  if (!features) {
    return std::nullopt;
  }

  const auto provided = features.get(name);
  if (!provided) {
    return std::nullopt;
  }

  return Feature(name, provided);
}

auto mlir::ktdf_arch::getFeature(Operation* op, StringAttr name)
    -> std::optional<Feature> {
  const auto features = getProperty<FeaturesAttr>(op);
  if (!features) {
    return std::nullopt;
  }

  const auto provided = features.get(name);
  if (!provided) {
    return std::nullopt;
  }

  return Feature(name, provided);
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchOpInterfaces.cpp.inc"
