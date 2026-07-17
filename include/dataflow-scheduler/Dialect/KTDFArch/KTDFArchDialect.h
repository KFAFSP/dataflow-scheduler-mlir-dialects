//===-- KTDFArchDialect.h ---------------------------------------*- c++ -*-===//
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
// This file defines the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectInterface.h>

namespace mlir::ktdf_arch {

/// Represents a provided or required architectural feature.
struct Feature : NamedAttribute {
  /*implicit*/ Feature(const NamedAttribute& base) : NamedAttribute(base) {}
  using NamedAttribute::NamedAttribute;

  /// Verifies the usage of the feature on @p op .
  auto verify(Operation* op) const -> LogicalResult;

  /// Determines whether @p provided fulfills this requirement.
  auto matchedBy(Attribute provided) const -> bool;
  /// Determines whether @p op fulfills this requirement.
  auto matchedBy(Operation* op) const -> bool;
};

/// Interface for dialects which define architectural features.
struct FeatureDialectInterface
    : DialectInterface::Base<FeatureDialectInterface> {
  explicit FeatureDialectInterface(Dialect* dialect) : Base(dialect) {}

  /// Verifies the usage of @p feature on @p op .
  ///
  /// If @p op is a `ktdf_arch` operation, then the feature is a provider
  /// declaration. Otherwise, the feature is a consumer constraint.
  virtual auto verify(Operation* op, const Feature& feature) const
      -> LogicalResult = 0;

  /// Determines whether @p required is fulfilled by @p provided .
  ///
  /// The @p provided attribute is a valid value of the feature with the same
  /// name as the @p required feature. If both attribute values are equal, the
  /// result must be `true`. The default implementation tests only this.
  [[nodiscard]] virtual auto test(Attribute provided,
                                  const Feature& required) const -> bool {
    return required.getValue() == provided;
  }
};

class KTDFArchDialect;

using AttrVerifyFn = LogicalResult (*)(Operation*, const NamedAttribute&);
using AttrVerifyMap = llvm::DenseMap<StringAttr, AttrVerifyFn>;
using FeatureTestFn = bool (*)(Attribute, const Feature&);
using FeatureTestMap = llvm::DenseMap<StringAttr, FeatureTestFn>;

}  // namespace mlir::ktdf_arch

/// Auto-generated includes.
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h.inc"  // IWYU pragma: export

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_H_