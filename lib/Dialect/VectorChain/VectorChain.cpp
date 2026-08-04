//===-- VectorChain.cpp ------------------------------------------*- c++ -*-==//
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
// This file implements the vectorchain dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"

using namespace mlir;
using namespace mlir::vectorchain;

//===----------------------------------------------------------------------===//
// VectorChain Enums
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChainEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorChainDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/VectorChain/VectorChainDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/VectorChain/VectorChainAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.cpp.inc"

void VectorChainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/VectorChain/VectorChainAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantBitstreamOp
//===----------------------------------------------------------------------===//

ParseResult ConstantBitstreamOp::parse(OpAsmParser& parser,
                                       OperationState& state) {
  // Parse the attributes and the result type.
  Type op_type;
  auto op_result = parser.parseOptionalAttrDict(state.attributes) ||
                   parser.parseColonType(op_type) ||
                   parser.addTypeToList(op_type, state.types);

  return failure(op_result);
}

void ConstantBitstreamOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  auto is_symbol_attr = op->getAttr("is_symbol");
  bool convert_values =
      !is_symbol_attr ||
      mlir::cast<BoolAttr>(is_symbol_attr).getValue() == false;

  if (convert_values) {
    p << " {value = [";

    // Print each of the interpreted_value elements as hex.
    auto values = op.getValue().getValue();
    unsigned values_size = values.size();
    for (unsigned i = 0; i < values_size; ++i) {
      auto int_attr = mlir::cast<IntegerAttr>(values[i]);
      int64_t int_val = int_attr.getInt();
      p << "0x" << llvm::Twine::utohexstr((uint64_t)int_val);
      if (i < values_size - 1) p << ", ";
    }

    p << "]}";
  } else {
    p.printOptionalAttrDict(op->getAttrs());
  }

  // Print the return type.
  p << " : " << op.getType();
}

LogicalResult ConstantBitstreamOp::verify() {
  auto& op = *this;

  // The number of elements in the interpreted_value array should match the
  // number of elements in the result type. The result type is expected to be a
  // vector.
  int value_size = op.getValue().size();
  auto result_type = op.getResult().getType();
  mlir::Type elements_type;
  int num_of_elements = 0;
  unsigned result_bitwidth = 0;
  bool is_int_or_float = false;
  if (auto const_vtype = mlir::dyn_cast<VectorType>(result_type)) {
    elements_type = const_vtype.getElementType();
    num_of_elements = const_vtype.getNumElements();
    is_int_or_float = elements_type.isIntOrFloat();
    result_bitwidth = elements_type.getIntOrFloatBitWidth();
  } else if (auto const_customvtype =
                 mlir::dyn_cast<dataflow::CustomVectorType>(result_type)) {
    elements_type = const_customvtype.getElementType();
    num_of_elements = const_customvtype.getNumElements();
    // Must call utilities since elements_type may or may not be a custom
    // dataflow type.
    is_int_or_float = dataflow::isIntOrFloatType(elements_type);
    result_bitwidth = dataflow::getIntOrFloatBitWidth(elements_type);
  } else {
    op->emitOpError("result element type is not vector type.");
    return failure();
  }
  if (!is_int_or_float) {
    op->emitOpError("result element type is not ints or floats");
    return failure();
  }

  if (num_of_elements != value_size) {
    op->emitOpError("result does not match value array");
    return failure();
  }

  auto is_symbol_attr = op->getAttr("is_symbol");
  bool convert_values =
      !is_symbol_attr ||
      mlir::cast<BoolAttr>(is_symbol_attr).getValue() == false;
  if (convert_values) {
    // The bitwidths of the elements of the value attribute should not exceed
    // the bitwidth of the output vector element type.
    auto values = op.getValue().getValue();
    for (int i = 0; i < value_size; ++i) {
      auto int_attr = mlir::dyn_cast<IntegerAttr>(values[i]);
      if (!int_attr) {
        op->emitOpError(
            "value attribute should only contain integer representations");
        return failure();
      }

      int64_t int_val = int_attr.getInt();
      int64_t mask = pow(2, result_bitwidth) - 1;
      if ((int_val & mask) != int_val) {
        op->emitOpError(
            "value attribute contains element that exceeds bitwidth of output "
            "element type");
        return failure();
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto& op = *this;

  // The number of elements in the result type should be equivalent to the
  // number of elements in the input type.
  int result_size = dataflow::getNumElements(op.getResult().getType());
  int input_size = dataflow::getNumElements(op.getInput().getType());
  if (result_size != input_size) {
    op->emitOpError("result size does not match input size");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleOp::verify() {
  auto& op = *this;

  // The number of elements in the result type should be equivalent to the
  // product of the number of elements in the indices and the number of
  // repetitions.
  auto indices = op.getIndices();
  auto indices_size = indices.size();
  const auto num_elements =
      static_cast<std::size_t>(dataflow::getNumElements(getResult().getType()));
  if (num_elements != indices_size * op.getRepetition()) {
    op->emitOpError("result does not scale with indices and repetitions");
    return failure();
  }

  const auto input_elements_type = dataflow::getElementType(op.getInput().getType());
  const auto input_num_of_elements = dataflow::getNumElements(op.getInput().getType());

  // The element type of the input should match the type of the output (type
  // only, not necessarily number of elements).
  const auto element_type = dataflow::getElementType(getResult().getType());
  if (element_type != input_elements_type) {
    op->emitOpError(
        "input element type does not match output element "
        "type");
    return failure();
  }
  // The elements referred to in the indices should not refer to out of bounds
  // elements from the input. Negative element values are only valid if a
  // corresponding variable or pad operand was provided.
  int max_element = input_num_of_elements - 1;
  int min_element = -(static_cast<int>(op.getVariable().size()) +
                      static_cast<int>(op.getPad().size()));
  for (std::size_t i = 0; i < indices_size; ++i) {
    auto element = dyn_cast<IntegerAttr>(indices[i]);
    if (!element) {
      op->emitOpError("indices element is not integer");
      return failure();
    }

    int64_t element_value = element.getInt();
    if (element_value < min_element || element_value > max_element) {
      op->emitOpError("out of bounds indices element");
      return failure();
    }
  }

  return success();
}

ParseResult ShuffleOp::parse(OpAsmParser& parser, OperationState& state) {
  OpAsmParser::UnresolvedOperand input;
  SmallVector<OpAsmParser::UnresolvedOperand> variableOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> padOperands;
  OpAsmParser::UnresolvedOperand mask;

  Type inputType, maskType, resultType;
  SmallVector<Type> variableTypes;
  SmallVector<Type> padTypes;

  if (parser.parseKeyword("input") || parser.parseLParen() ||
      parser.parseOperand(input) || parser.parseRParen())
    return failure();

  bool hasVariable = false;
  bool hasPad = false;
  bool hasMask = false;

  while (succeeded(parser.parseOptionalComma())) {
    if (!hasVariable && succeeded(parser.parseOptionalKeyword("variable"))) {
      hasVariable = true;
      if (parser.parseLParen() ||
          parser.parseOperandList(variableOperands,
                                  OpAsmParser::Delimiter::None) ||
          parser.parseRParen())
        return failure();
      continue;
    }
    if (!hasPad && succeeded(parser.parseOptionalKeyword("pad"))) {
      hasPad = true;
      if (parser.parseLParen() ||
          parser.parseOperandList(padOperands,
                                  OpAsmParser::Delimiter::None) ||
          parser.parseRParen())
        return failure();
      continue;
    }
    if (!hasMask && succeeded(parser.parseOptionalKeyword("mask"))) {
      hasMask = true;
      if (parser.parseLParen() || parser.parseOperand(mask) ||
          parser.parseRParen())
        return failure();
      continue;
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'variable', 'pad', or 'mask' keyword");
  }

  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  if (parser.parseColon() || parser.parseType(inputType))
    return failure();

  if (hasVariable) {
    for (size_t i = 0; i < variableOperands.size(); ++i) {
      Type varType;
      if (parser.parseComma() || parser.parseType(varType))
        return failure();
      variableTypes.push_back(varType);
    }
  }
  if (hasPad) {
    for (size_t i = 0; i < padOperands.size(); ++i) {
      Type padType;
      if (parser.parseComma() || parser.parseType(padType))
        return failure();
      padTypes.push_back(padType);
    }
  }
  if (hasMask) {
    if (parser.parseComma() || parser.parseType(maskType))
      return failure();
  }
  if (parser.parseComma() || parser.parseType(resultType))
    return failure();

  if (parser.resolveOperand(input, inputType, state.operands))
    return failure();
  if (hasVariable) {
    for (size_t i = 0; i < variableOperands.size(); ++i)
      if (parser.resolveOperand(variableOperands[i], variableTypes[i],
                                state.operands))
        return failure();
  }
  if (hasPad) {
    for (size_t i = 0; i < padOperands.size(); ++i)
      if (parser.resolveOperand(padOperands[i], padTypes[i], state.operands))
        return failure();
  }
  if (hasMask) {
    if (parser.resolveOperand(mask, maskType, state.operands))
      return failure();
  }

  auto& builder = parser.getBuilder();
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(variableOperands.size()),
           static_cast<int32_t>(padOperands.size()), hasMask ? 1 : 0}));
  state.addTypes(resultType);
  return success();
}

void ShuffleOp::print(OpAsmPrinter& p) {
  Operation* op = getOperation();
  auto segmentSizes =
      op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  int32_t numVariable = segmentSizes.asArrayRef()[1];
  int32_t numPad = segmentSizes.asArrayRef()[2];
  int32_t numMask = segmentSizes.asArrayRef()[3];

  auto operands = op->getOperands();
  unsigned operandIdx = 0;

  p << " input(" << operands[operandIdx++] << ")";
  if (numVariable > 0) {
    p << ", variable(";
    for (int32_t i = 0; i < numVariable; ++i) {
      if (i > 0) p << ", ";
      p << operands[operandIdx++];
    }
    p << ")";
  }
  if (numPad > 0) {
    p << ", pad(";
    for (int32_t i = 0; i < numPad; ++i) {
      if (i > 0) p << ", ";
      p << operands[operandIdx++];
    }
    p << ")";
  }
  if (numMask > 0)
    p << ", mask(" << operands[operandIdx++] << ")";

  SmallVector<StringRef> elidedAttrs = {"operandSegmentSizes"};
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  operandIdx = 0;
  p << " : " << operands[operandIdx++].getType();
  for (int32_t i = 0; i < numVariable; ++i)
    p << ", " << operands[operandIdx++].getType();
  for (int32_t i = 0; i < numPad; ++i)
    p << ", " << operands[operandIdx++].getType();
  if (numMask > 0)
    p << ", " << operands[operandIdx++].getType();
  p << ", " << op->getResult(0).getType();
}

void ShuffleOp::build(OpBuilder& builder, OperationState& state,
                      Type resultType, Value input, ValueRange variable,
                      ValueRange pad, Value mask, ArrayAttr indices,
                      IntegerAttr repetition, StringAttr dbgName) {
  state.addOperands(input);
  state.addOperands(variable);
  state.addOperands(pad);
  if (mask) state.addOperands(mask);
  if (dbgName) state.addAttribute("dbgName", dbgName);
  state.addAttribute("indices", indices);
  state.addAttribute("repetition", repetition);
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(variable.size()),
           static_cast<int32_t>(pad.size()), mask ? 1 : 0}));
  state.addTypes(resultType);
}

void ShuffleOp::build(OpBuilder& builder, OperationState& state,
                      Type resultType, Value input, ArrayAttr indices,
                      IntegerAttr repetition, StringAttr dbgName) {
  state.addOperands(input);
  if (dbgName) state.addAttribute("dbgName", dbgName);
  state.addAttribute("indices", indices);
  state.addAttribute("repetition", repetition);
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr({1, 0, 0, 0}));
  state.addTypes(resultType);
}

void ShuffleOp::build(OpBuilder& builder, OperationState& state,
                      Type resultType, Value input, Value mask,
                      ArrayAttr indices, IntegerAttr repetition,
                      StringAttr dbgName) {
  state.addOperands(input);
  if (mask) state.addOperands(mask);
  if (dbgName) state.addAttribute("dbgName", dbgName);
  state.addAttribute("indices", indices);
  state.addAttribute("repetition", repetition);
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({1, 0, 0, mask ? 1 : 0}));
  state.addTypes(resultType);
}

void ShuffleOp::build(OpBuilder& builder, OperationState& state,
                      Type resultType, Value input, ValueRange variable,
                      ValueRange pad, ArrayAttr indices,
                      IntegerAttr repetition, StringAttr dbgName) {
  state.addOperands(input);
  state.addOperands(variable);
  state.addOperands(pad);
  if (dbgName) state.addAttribute("dbgName", dbgName);
  state.addAttribute("indices", indices);
  state.addAttribute("repetition", repetition);
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(variable.size()),
           static_cast<int32_t>(pad.size()), 0}));
  state.addTypes(resultType);
}
