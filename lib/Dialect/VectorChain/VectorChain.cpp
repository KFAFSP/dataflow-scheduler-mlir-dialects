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
  // elements from the input. Element values of -1 are still valid as these are
  // used to indicate padding.
  int max_element = input_num_of_elements - 1;
  for (std::size_t i = 0; i < indices_size; ++i) {
    auto element = dyn_cast<IntegerAttr>(indices[i]);
    if (!element) {
      op->emitOpError("indices element is not integer");
      return failure();
    }

    int64_t element_value = element.getInt();
    if (element_value < -1 || element_value > max_element) {
      op->emitOpError("out of bounds indices element");
      return failure();
    }
  }

  return success();
}
