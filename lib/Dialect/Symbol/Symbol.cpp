//===-- Symbol.cpp -----------------------------------------------*- c++ -*-==//
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
// This file implements the symbol dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::symbol;

namespace {

auto parseImmutableMap(OpAsmParser& parser,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand>& keys,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand>& values)
    -> ParseResult {
  return parser.parseCommaSeparatedList(
      AsmParser::Delimiter::Paren, [&]() -> ParseResult {
        return failure(
            parser.parseLSquare() || parser.parseOperand(keys.emplace_back()) ||
            parser.parseArrow() || parser.parseOperand(values.emplace_back()) ||
            parser.parseRSquare());
      });
}

void printImmutableMap(OpAsmPrinter& printer, Operation* /*op*/, ValueRange keys,
                       ValueRange values) {
  printer << "(";
  llvm::interleaveComma(llvm::zip_equal(keys, values), printer, [&](auto pair) {
    printer << "[" << std::get<0>(pair) << " -> " << std::get<1>(pair) << "]";
  });
  printer << ")";
}

}  // namespace

//===----------------------------------------------------------------------===//
// SymbolDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Symbol/SymbolDialect.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Symbol/Symbol.cpp.inc"

void SymbolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Symbol/Symbol.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SymbolImmutableMappingOp
//===----------------------------------------------------------------------===//

LogicalResult SymbolImmutableMappingOp::verify() {
  auto& op = *this;
  if (op.getKeys().size() != op.getValues().size()) {
    op.emitError("Mismatching keys/values sizes in SymbolImmutableMapping\n");
    return failure();
  }

  for (auto key : op.getKeys()) {
    Operation* key_def = key.getDefiningOp();
    if (!key_def || !key_def->hasTrait<mlir::OpTrait::ConstantLike>()) {
      op.emitError() << "All keys expected to be constant operations\n";
      return failure();
    }
  }

  for (auto value : op.getValues()) {
    if (!isa<symbol::CreateSymbolOp>(value.getDefiningOp())) {
      op.emitError(
          "Values in SymbolImmutableMapping should be results of "
          "symbol.create_symbol\n");
      return failure();
    }
    OpBuilder builder(op);
    if (value.getType() != builder.getIndexType()) {
      op.emitError(
          "Values in SymbolImmutableMapping expected to have Index type\n");
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolQueryMapOp
//===----------------------------------------------------------------------===//

LogicalResult SymbolQueryMapOp::verify() {
  auto& op = *this;
  // Key expected to be a loop iterator argument or a Value.
  if (dyn_cast<BlockArgument>(op.getKey())) {
    auto parent_op = op.getKey().getParentRegion()->getParentOp();
    // Accept any loop-like op (i.e. one implementing LoopLikeOpInterface).
    if (!isa<mlir::LoopLikeOpInterface>(parent_op)) {
      op.emitError("key's parentOp must be a ForOp.");
      return failure();
    }
  }
  if (!op.getMap().getDefiningOp<SymbolImmutableMappingOp>()) {
    op.emitError("map's defOp must be symbol_immutable_mapping op.");
    return failure();
  }
  return success();
}
