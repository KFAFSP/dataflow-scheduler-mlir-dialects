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
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"  // IWYU pragma: keep

/// Auto-generated includes.
#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h.inc"  // IWYU pragma: export

namespace mlir::ktdf_arch {

/// Parses a `ktdf_arch` type mnemonic, or falls back to a qualified type.
///
/// Note that builtin type mnemonics can not be parsed with this function.
auto parseShortType(OpAsmParser& parser, Type& type) -> ParseResult;

/// Prints a `ktdf_arch` type mnemonic, or falls back to a qualified type.
///
/// @pre @p type is not a builtin type.
void printShortType(OpAsmPrinter& printer, Operation* op, Type type);

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_H_
