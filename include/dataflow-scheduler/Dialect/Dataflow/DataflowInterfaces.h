//===-- DataflowInterfaces.h ------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_H_
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>

#include <type_traits>

namespace mlir::dataflow::detail {

template <class T>
using dbgName_property_t =
    decltype(std::declval<typename T::Properties>().dbgName);

template <class T, class = void>
struct has_dbgName_property : std::false_type {};

template <class T>
struct has_dbgName_property<T, std::void_t<dbgName_property_t<T>>>
    : std::integral_constant<
          bool, std::is_same_v<dbgName_property_t<T>, mlir::StringAttr>> {};

template <class T>
static constexpr auto has_dbgName_property_v = has_dbgName_property<T>::value;

}  // namespace mlir::dataflow::detail

/// Auto-generated includes.
#include "dataflow-scheduler/Dialect/Dataflow/DataflowOpInterfaces.h.inc"  // IWYU pragma: export

namespace mlir::dataflow {

/// Gets the optional debug name attribute of @p op .
template <class Op>
auto getDbgNameAttr(Op op) -> std::enable_if_t<
    std::is_base_of_v<detail::DebugNameOpInterfaceTrait<Op>, Op>, StringAttr> {
  return op.getProperties().dbgName;
}

/// Gets the optional debug name attribute of @p op .
///
/// Falls back to (discardable) attribute manipulation if the op does not
/// implement the `DebugNameOpInterface`.
auto getDbgNameAttr(Operation* op) -> StringAttr;

/// Gets the optional debug name of @p op .
///
/// Falls back to (discardable) attribute manipulation if the op does not
/// implement the `DebugNameOpInterface`.
template <class Op>
auto getDbgName(Op op) -> std::optional<StringRef> {
  if (const auto attr = getDbgNameAttr(op); attr) {
    return attr.getValue();
  }

  return std::nullopt;
}

/// Sets the optional debug name attribute of @p op to @p name (or removes it).
template <class Op>
auto setDbgNameAttr(Op op, StringAttr name) -> std::enable_if_t<
    std::is_base_of_v<detail::DebugNameOpInterfaceTrait<Op>, Op>, void> {
  op.getProperties().dbgName = name;
}

/// Sets the optional debug name attribute of @p op to @p name (or removes it).
///
/// Falls back to (discardable) attribute manipulation if the op does not
/// implement the `DebugNameOpInterface`.
void setDbgNameAttr(Operation* op, StringAttr name);

/// Sets the optional debug name of @p op to @p name (or removes it).
///
/// Falls back to (discardable) attribute manipulation if the op does not
/// implement the `DebugNameOpInterface`.
template <class Op>
auto setDbgName(Op op, std::optional<StringRef> name) {
  MLIRContext* ctx;
  if constexpr (std::is_pointer_v<Op>) {
    ctx = op->getContext();
  } else {
    ctx = op.getContext();
  }

  setDbgNameAttr(op, name ? StringAttr::get(ctx, name.value()) : nullptr);
}

}  // namespace mlir::dataflow

#endif  // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_H_