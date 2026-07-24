//===-- DataflowTypes.cpp ----------------------------------------*- c++ -*-==//
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

#include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>

#include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// CustomVectorType
//===----------------------------------------------------------------------===//

auto CustomVectorType::parse(AsmParser& parser) -> Type {
  std::size_t num_elements;
  Type element_type;
  if (parser.parseLess() || parser.parseInteger(num_elements) ||
      parser.parseXInDimensionList() || parser.parseType(element_type) ||
      parser.parseGreater()) {
    return nullptr;
  }

  return CustomVectorType::get(element_type, num_elements);
}

void CustomVectorType::print(AsmPrinter& printer) const {
  printer << "<" << getNumElements() << "x" << getElementType() << ">";
}

//===----------------------------------------------------------------------===//
// DataflowDialect
//===----------------------------------------------------------------------===//

namespace {

auto parseMXMnemonic(AsmParser& parser, StringRef& mnemonic, Type& result)
    -> OptionalParseResult {
  return AsmParser::KeywordSwitch<OptionalParseResult>(parser)
      .Case("mxfp4",
            [&](llvm::StringRef, llvm::SMLoc) {
              result = CustomMXFloatType::get(parser.getContext(), 4U);
              return llvm::success();
            })
      .Case("mxfp8",
            [&](llvm::StringRef, llvm::SMLoc) {
              result = CustomMXFloatType::get(parser.getContext(), 8U);
              return llvm::success();
            })
      .Case("mxint4",
            [&](llvm::StringRef, llvm::SMLoc) {
              result = CustomMXIntType::get(parser.getContext(), 4U);
              return llvm::success();
            })
      .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
        mnemonic = keyword;
        return std::nullopt;
      });
}

auto printCustomMXType(AsmPrinter& printer, Type type) -> LogicalResult {
  return llvm::TypeSwitch<Type, LogicalResult>(type)
      .Case([&](CustomMXFloatType custom) {
        switch (custom.getWidth()) {
          case 4U:
            printer << "custom<mxfp4>";
            return success();
          case 8U:
            printer << "custom<mxfp8>";
            return success();
          default:
            return failure();
        }
      })
      .Case([&](CustomMXIntType custom) {
        switch (custom.getWidth()) {
          case 4U:
            printer << "custom<mxint4>";
            return success();
          default:
            return failure();
        }
      })
      .Default(failure());
}

}  // namespace

auto DataflowDialect::parseType(DialectAsmParser& parser) const -> Type {
  auto loc = parser.getCurrentLocation();

  StringRef mnemonic;
  Type result;
  if (generatedTypeParser(parser, &mnemonic, result).has_value()) {
    return result;
  }

  if (mnemonic != "custom") {
    parser.emitError(loc, "expected mnemonic or 'custom'");
    return nullptr;
  }
  if (parser.parseLess()) {
    return nullptr;
  }

  if (parseMXMnemonic(parser, mnemonic, result).has_value()) {
    if (parser.parseGreater()) {
      return nullptr;
    }
    return result;
  }

  parser.emitError(loc, "expected 'mxfp4', 'mxfp8' or 'mxint4' but got '")
      << mnemonic << "'";
  return nullptr;
}

void DataflowDialect::printType(Type type, DialectAsmPrinter& printer) const {
  if (succeeded(printCustomMXType(printer, type))) {
    return;
  }

  if (succeeded(generatedTypePrinter(type, printer))) {
    return;
  }

  llvm_unreachable("unhandled dataflow dialect type");
}

void DataflowDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.cpp.inc"
      >();
}
