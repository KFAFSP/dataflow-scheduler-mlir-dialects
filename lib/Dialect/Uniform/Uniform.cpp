//===-- Uniform.cpp ----------------------------------------------*- c++ -*-==//
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
// This file implements the uniform dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/OpTraits.h"

using namespace mlir;
using namespace mlir::uniform;

//===----------------------------------------------------------------------===//
// UniformDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Uniform/UniformDialect.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Uniform/Uniform.cpp.inc"

void UniformDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Uniform/Uniform.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DefImmutableMappingOp
//===----------------------------------------------------------------------===//

// syntax:%result = uniform.def_immutable_mapping([%1 -> %2],[%3 -> %4]):index
ParseResult DefImmutableMappingOp::parse(OpAsmParser& parser,
                                         OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> keys, values;
  OpAsmParser::UnresolvedOperand key, value;
  auto op_result = parser.parseLParen().failed();
  while (parser.parseOptionalLSquare().succeeded()) {
    op_result = op_result || parser.parseOperand(key) || parser.parseArrow() ||
                parser.parseOperand(value) || parser.parseRSquare();
    keys.push_back(key);
    values.push_back(value);
    if (parser.parseOptionalComma().failed())
      break;
  }
  Type result_type;
  op_result =
      op_result || parser.parseRParen() || parser.parseColonType(result_type);

  Type value_type;
  if (parser.parseOptionalComma().succeeded()) {
    op_result = op_result || parser.parseType(value_type);
  } else {
    value_type = builder.getIndexType();
  }

  result.types.push_back(result_type);
  int size = keys.size();
  result.attributes.push_back(
      builder.getNamedAttr(getOperandSegmentSizesAttrName(OperationName(
                               getOperationName(), builder.getContext())),
                           builder.getDenseI32ArrayAttr({size, size})));
  op_result = op_result ||
              parser.resolveOperands(keys, index_type, result.operands) ||
              parser.resolveOperands(values, value_type, result.operands);

  return failure(op_result);
}

void DefImmutableMappingOp::print(OpAsmPrinter& p) {
  auto& op = *this;

  Type value_type = op.getType();
  int size = op.getKeys().size();
  p << '(';
  for (int i = 0; i < size; i++) {
    p << '[';
    p.printOperand(op.getKeys()[i]);
    p << " -> ";
    value_type = op.getValues()[i].getType();
    p.printOperand(op.getValues()[i]);
    p << ']';
    if (i < size - 1) p << ", ";
  }
  p << "):";
  p.printType(op.getResult().getType());

  if (value_type != op.getResult().getType()) {
    p << ", ";
    p.printType(value_type);
  }
}

LogicalResult DefImmutableMappingOp::verify() {
  auto& op = *this;
  auto key0 = op.getKeys()[0].getDefiningOp<dataflow::GetUnitOp>();
  if (key0) {
    for (auto key : op.getKeys()) {
      if (!isa<dataflow::GetUnitOp>(key.getDefiningOp())) {
        op.emitError("All Keys are not of same type\n");
        return failure();
      }
    }
  }
  auto I = op.getValues().begin(), E = op.getValues().end();
  if (I != E) {
    auto firstValDef = (*I++).getDefiningOp();
    if (firstValDef->hasTrait<scheduler::OpTrait::ScalarValue>()) {
      // A mix of scalar op value such as scalar_add, scalar_sub may appear as
      // map values. The scalar values will need to resolve to compile-time
      // constants but verification is only confirming homogeneity or potential
      // for the map values to resolve to scalar constants.
      for (; I != E; ++I) {
        if (!(*I).getDefiningOp()
                 ->hasTrait<scheduler::OpTrait::ScalarValue>()) {
          op.emitError("All map values are expected to be scalar values.");
          return failure();
        }
      }
    } else {
      // If the map values are not scalar values then homogeneity is determined
      // by the specific op.
      for (; I != E; ++I) {
        if (firstValDef->getName() != (*I).getDefiningOp()->getName()) {
          op.emitError("All map values should be of same type.");
          return failure();
        }
      }
    }
  }
  return success();
}

std::optional<Value> DefImmutableMappingOp::getValue(Value key) {
  auto& op = *this;

  int size = op.getKeys().size();

  for (int i = 0; i < size; i++) {
    if (key == op.getKeys()[i]) {
      return op.getValues()[i];
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// QueryMapOp
//===----------------------------------------------------------------------===//

// syntax:%result = uniform.query_map(map:%2, key:%1) : index
ParseResult QueryMapOp::parse(OpAsmParser& parser, OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  OpAsmParser::UnresolvedOperand map, key;
  Type result_type;

  auto op_result = parser.parseLParen() || parser.parseKeyword("map") ||
                   parser.parseColon() || parser.parseOperand(map) ||
                   parser.parseComma() || parser.parseKeyword("key") ||
                   parser.parseColon() || parser.parseOperand(key) ||
                   parser.parseRParen() || parser.parseColonType(result_type);
  result.types.push_back(result_type);
  op_result = op_result ||
              parser.resolveOperand(map, index_type, result.operands) ||
              parser.resolveOperand(key, index_type, result.operands);

  return failure(op_result);
}

void QueryMapOp::print(OpAsmPrinter& p) {
  auto& op = *this;

  p << "(map:";
  p.printOperand(op.getMap());
  p << ", key:";
  p.printOperand(op.getKey());
  p << ") : ";
  p.printType(op.getResult().getType());
}

LogicalResult QueryMapOp::verify() { return success(); }
