//===-- KTDFArchInterfaces.h ------------------------------------*- c++ -*-===//
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
// This file defines the interfaces of the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_H_

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"

namespace mlir::ktdf_arch {

/// Verifies an operation that represents a generic resource.
auto verifyResource(Operation* op) -> LogicalResult;

/// Verifies an operation that represents a generic link.
auto verifyLink(Operation* op) -> LogicalResult;

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

/// Gets the intrinsic @p PropertyAttr of @p op , if it is set.
template <class PropertyAttr>
auto getProperty(Operation* op) -> PropertyAttr {
  const auto name = PropertyAttr::getAttrName(op->getContext());
  return cast_if_present<PropertyAttr>(op->getDiscardableAttr(name));
}

/// Sets the intrinsic @p PropertyAttr of @p op .
template <class PropertyAttr>
void setProperty(Operation* op, PropertyAttr attr) {
  const auto name = PropertyAttr::getAttrName(op->getContext());
  op->setDiscardableAttr(name, attr);
}

/// Removes the intrinsic @p PropertyAttr of @p op and returns it, if any.
template <class PropertyAttr>
auto removeProperty(Operation* op) -> PropertyAttr {
  const auto name = PropertyAttr::getAttrName(op->getContext());
  return cast_if_present<PropertyAttr>(op->removeDiscardableAttr(name));
}

//===----------------------------------------------------------------------===//
// Features
//===----------------------------------------------------------------------===//

/// Gets the Feature of @p op that satisfies @p required , if any.
[[nodiscard]] auto getFeature(Operation* op, const Feature& required)
    -> std::optional<Feature>;
/// Gets the Feature of @p op with @p name , if any.
[[nodiscard]] auto getFeature(Operation* op, StringRef name)
    -> std::optional<Feature>;
/// Gets the Feature of @p op with @p name , if any.
[[nodiscard]] auto getFeature(Operation* op, StringAttr name)
    -> std::optional<Feature>;
/// Gets the @p FeatureAttr of @p op , if any.
template <class FeatureAttr>
[[nodiscard]] auto getFeature(Operation* op) -> FeatureAttr {
  const auto name = FeatureAttr::getAttrName(op->getContext());
  if (const auto maybe_feature = getFeature(op, name); maybe_feature) {
    return dyn_cast<FeatureAttr>(maybe_feature->getValue());
  }

  return nullptr;
}
/// Gets the @p FeatureAttr of @p op that satisfies @p required , if any.
template <class FeatureAttr>
[[nodiscard]] auto getFeature(Operation* op, FeatureAttr required)
    -> std::enable_if_t<std::is_base_of_v<Attribute, FeatureAttr>,
                        FeatureAttr> {
  const auto provided = getFeature<FeatureAttr>(op);
  if (provided.test(required)) {
    return provided;
  }

  return nullptr;
}

}  // namespace mlir::ktdf_arch

/// Auto-generated includes.
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchOpInterfaces.h.inc"  // IWYU pragma: export

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_H_