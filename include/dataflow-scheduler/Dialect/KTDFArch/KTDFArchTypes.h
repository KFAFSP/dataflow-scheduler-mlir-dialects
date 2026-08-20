//===-- KTDFArchTypes.h -----------------------------------------*- c++ -*-===//
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
// This file defines the types in the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_H_

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"  // IWYU pragma: keep

namespace mlir::ktdf_arch {

struct EndpointType;

}  // namespace mlir::ktdf_arch

/// Auto-generated includes.
#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h.inc"  // IWYU pragma: export

namespace mlir::ktdf_arch {

/// Parses a `ktdf_arch` type mnemonic, or falls back to a qualified type.
///
/// Note that builtin type mnemonics can not be parsed with this function.
auto parseShortType(AsmParser& parser, Type& type) -> ParseResult;

/// Prints a `ktdf_arch` type mnemonic, or falls back to a qualified type.
///
/// @pre @p type is not a builtin type.
void printShortType(AsmPrinter& printer, Type type);
/// @copydoc printShortType(AsmPrinter&, Type)
void printShortType(OpAsmPrinter& printer, Operation* op, Type type);

//===----------------------------------------------------------------------===//
// EndpointType
//===----------------------------------------------------------------------===//

/// Named constraint for the EndpointType in TableGen.
struct EndpointType : Type {
  [[nodiscard]] static auto classof(Type type) -> bool {
    return isa<MemoryType, ExecutionUnitType, PortType>(type);
  }
  [[nodiscard]] static auto classof(MemoryType /*type*/) -> bool {
    return true;
  }
  [[nodiscard]] static auto classof(ExecutionUnitType /*type*/) -> bool {
    return true;
  }
  [[nodiscard]] static auto classof(PortType /*type*/) -> bool { return true; }

  using Type::Type;

  /*implicit*/ EndpointType(MemoryType type)
      : Type(static_cast<ImplType*>(type.getImpl())) {}
  /*implicit*/ EndpointType(ExecutionUnitType type)
      : Type(static_cast<ImplType*>(type.getImpl())) {}
  /*implicit*/ EndpointType(PortType type)
      : Type(static_cast<ImplType*>(type.getImpl())) {}
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_H_
