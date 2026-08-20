//===-- KTDFArchTypes.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// Custom Assembly Format Declarations
//===----------------------------------------------------------------------===//

namespace {

auto parseShortTypeTuple(AsmParser& parser, SmallVectorImpl<Type>& types)
    -> ParseResult {
  return parser.parseCommaSeparatedList(
      AsmParser::Delimiter::Paren, [&]() -> ParseResult {
        return parseShortType(parser, types.emplace_back());
      });
}

void printShortTypeTuple(AsmPrinter& printer, TypeRange types) {
  printer << "(";
  llvm::interleaveComma(types, printer,
                        [&](Type type) { printShortType(printer, type); });
  printer << ")";
}

}  // namespace

//===----------------------------------------------------------------------===//
// KTDFArchDialect
//===----------------------------------------------------------------------===//

void KTDFArchDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.cpp.inc"

auto mlir::ktdf_arch::parseShortType(AsmParser& parser, Type& type)
    -> ParseResult {
  if (auto result = parser.parseOptionalType(type); result.has_value()) {
    return result.value();
  }

  const auto loc = parser.getCurrentLocation();
  StringRef keyword{};
  if (const auto maybe_short = generatedTypeParser(parser, &keyword, type);
      maybe_short.has_value()) {
    if (maybe_short.value()) {
      return parser.emitError(loc, "expected 'ktdf_arch' type mnemonic");
    }

    return success();
  }

  return parser.emitError(loc, "expected 'ktdf_arch' type mnemonic, got '")
         << keyword << "'";
}

void mlir::ktdf_arch::printShortType(AsmPrinter& printer, Type type) {
  if (succeeded(generatedTypePrinter(type, printer))) {
    return;
  }

  assert(!isa<BuiltinDialect>(type.getDialect()));
  printer << type;
}

void mlir::ktdf_arch::printShortType(OpAsmPrinter& printer, Operation* /*op*/,
                                     Type type) {
  ktdf_arch::printShortType(printer, type);
}

//===----------------------------------------------------------------------===//
// NeighborhoodType
//===----------------------------------------------------------------------===//

auto NeighborhoodType::verify(function_ref<InFlightDiagnostic()> emit_error,
                              ArrayRef<Type> results,
                              ArrayRef<int64_t> dimensions) -> LogicalResult {
  for (auto [idx, type] : llvm::enumerate(results)) {
    if (!isa<EndpointType>(type)) {
      return emit_error() << "result #" << idx << " (" << type
                          << ") must be a resource endpoint type";
    }
  }

  for (auto [idx, dim] : llvm::enumerate(dimensions)) {
    if (dim <= 0) {
      return emit_error() << "dimension #" << idx << " (" << dim
                          << ") must be positive";
    }
  }

  return success();
}

auto NeighborhoodType::cloneWith(
    std::optional<ArrayRef<Type>> results,
    std::optional<ArrayRef<int64_t>> dimensions) const -> NeighborhoodType {
  if (!results && !dimensions) {
    return *this;
  }

  return get(getContext(), results.value_or(getResults()),
             dimensions.value_or(getDimensions()));
}
