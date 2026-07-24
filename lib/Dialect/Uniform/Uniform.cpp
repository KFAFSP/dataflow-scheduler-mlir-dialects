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
#include <mlir/Support/LLVM.h>

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
// UniformizeRegionsOp
//===----------------------------------------------------------------------===//

// syntax:
// %results = uniform.uniformize_regions ->(index) {
//   (%arg0 -> %1, %2) {
//     Region1
//   }
//   (%arg1 -> %3) {
//     Region2
//   }
// }
ParseResult UniformizeRegionsOp::parse(OpAsmParser& parser,
                                       OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> unit_list;
  OpAsmParser::Argument region_arg;
  region_arg.type = index_type;
  SmallVector<int32_t, 1> list_sizes;
  auto* body = result.addRegion();

  auto op_result =
      parser.parseOptionalArrowTypeList(result.types) || parser.parseLBrace();

  while (parser.parseOptionalLParen().succeeded()) {
    op_result = op_result || parser.parseArgument(region_arg) ||
                parser.parseArrow() || parser.parseOperandList(unit_list) ||
                parser.parseRParen() || parser.parseRegion(*body, {region_arg});
    list_sizes.push_back(unit_list.size());
    op_result = op_result ||
                parser.resolveOperands(unit_list, index_type, result.operands);
    unit_list.clear();
    UniformizeRegionsOp::ensureTerminator(*body, builder, result.location);
    body = result.addRegion();
  }
  result.regions.pop_back();  // pop the empty region
  op_result = op_result || parser.parseOptionalRBrace() ||
              parser.parseOptionalAttrDict(result.attributes);
  result.addAttribute(getListSizesAttrStrName(),
                      builder.getI32ArrayAttr(list_sizes));

  return failure(op_result);
}

void UniformizeRegionsOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  auto num_regions = op.getRegions().size();

  p.printArrowTypeList(op.getResultTypes());

  p << " {";
  p.increaseIndent();
  for (std::size_t i = 0; i < num_regions; i++) {
    p.printNewline();
    p << '(' << op.getRegionArg(i) << " -> ";
    p.printOperands(op.getRegionUnitList(i));
    p << ')';
    p.printRegion(op.getRegion(i), false, true);
  }
  p.decreaseIndent();
  p.printNewline();
  p << '}';

  // Print all non-hidden attributes
  llvm::SmallVector<StringRef, 3> named_attr = {"list_sizes", "newly_added"};
  p.printOptionalAttrDict(op->getAttrs(), named_attr);
}

LogicalResult UniformizeRegionsOp::verify() {
  auto& op = *this;
  assert(op.getRegions().size() >= 1);
  assert(op.getRegions().size() == op.getListSizes().size());

  // check unit sizes are the same as list of sizes
  int size = op.getUnits().size();
  int accu_size = 0;
  for (std::size_t i = 0; i < op.getListSizes().size(); i++) {
    accu_size +=
        mlir::cast<IntegerAttr>(op.getListSizes().getValue()[i]).getInt();
  }
  assert((accu_size == size) &&
         "unit size has to be matched to the list sizes contents.");

  // no duplication in units
  llvm::DenseSet<mlir::Value> units;
  for (auto unit : op.getUnits()) {
    if (units.contains(unit)) {
      op.emitError("duplicated unit SSAs found!");
      return failure();
    } else {
      units.insert(unit);
    }
  }

  // verify units of unit_lists are either get_unit or create_group
  for (std::size_t idx = 0; idx < op.getRegions().size(); idx++) {
    auto units = op.getRegionUnitList(idx);
    if (units.size() == 1) {
      auto def_op = units.front().getDefiningOp();
      if (!isa<dataflow::GetUnitOp, dataflow::CreateGroupOp>(def_op)) {
        op.emitError(
            "region's unit_list must be programUnitOp or createGroupOp type!");
        return failure();
      }
    } else {
      for (auto unit : units) {
        if (!isa<dataflow::GetUnitOp>(unit.getDefiningOp())) {
          op.emitError(
              "when unit_list size > 1, all units must be programUnitOp!");
          return failure();
        }
      }
    }
  }

  // yieldOp's operand sizes are equal to results' size.
  const auto result_size = op.getResults().size();
  for (auto& region : op.getRegions()) {
    if (region.getBlocks().front().getTerminator()->getNumOperands() !=
        result_size) {
      op.emitError(
          "The size of the operands in the yieldOp of a uniformize_regionsOp "
          "region differs from the size of its results.");
      return failure();
    }
  }
  return success();
}

ValueRange UniformizeRegionsOp::getRegionUnitList(int pos) {
  int drop = 0;
  int keep =
      mlir::cast<IntegerAttr>(this->getListSizes().getValue()[pos]).getInt();
  for (auto idx = 0; idx < pos; idx++) {
    drop +=
        mlir::cast<IntegerAttr>(this->getListSizes().getValue()[idx]).getInt();
  }
  return this->getUnits().slice(drop, keep);
}

std::optional<ValueRange> UniformizeRegionsOp::getRegionUnitList(Value arg) {
  int size = this->getListSizes().size();
  for (int idx = 0; idx < size; idx++) {
    if (arg == getRegionArg(idx)) {
      return getRegionUnitList(idx);
    }
  }
  return std::nullopt;
}

std::optional<Region*> UniformizeRegionsOp::getSpecificRegion(Value arg) {
  int size = this->getListSizes().size();
  for (int idx = 0; idx < size; idx++) {
    if (arg == getRegionArg(idx)) {
      return &getRegion(idx);
    }
  }
  return std::nullopt;
}

Region* UniformizeRegionsOp::getRegionFromUnit(Value unit) {
  int idx_in_unit_list = 0;
  for (auto get_unit : this->getUnits()) {
    if (get_unit == unit) break;
    ++idx_in_unit_list;
  }

  int num_units_visited = 0;
  auto list_sizes = this->getListSizes();
  for (std::size_t i = 0; i < list_sizes.size(); ++i) {
    num_units_visited += mlir::cast<IntegerAttr>(list_sizes[i]).getInt();
    if (num_units_visited > idx_in_unit_list) return &this->getRegion(i);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// EqualizePatternOp
//===----------------------------------------------------------------------===//

// syntax:
// uniform.equalize_pattern
//   (%arg0 -> %1, %2) {
//     Region1
//   }
//   (%arg1 -> %3) {
//     Region2
//   }
// }
ParseResult EqualizePatternOp::parse(OpAsmParser& parser,
                                     OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  SmallVector<OpAsmParser::UnresolvedOperand, 1> unit_list;
  OpAsmParser::Argument region_arg;
  region_arg.type = index_type;
  SmallVector<int32_t, 1> list_sizes;
  auto* body = result.addRegion();

  auto op_result = parser.parseLBrace().failed();

  while (parser.parseOptionalLParen().succeeded()) {
    op_result = op_result || parser.parseArgument(region_arg) ||
                parser.parseArrow() || parser.parseOperandList(unit_list) ||
                parser.parseRParen() || parser.parseRegion(*body, {region_arg});
    list_sizes.push_back(unit_list.size());
    op_result = op_result ||
                parser.resolveOperands(unit_list, index_type, result.operands);
    unit_list.clear();
    UniformizeRegionsOp::ensureTerminator(*body, builder, result.location);
    body = result.addRegion();
  }
  result.regions.pop_back();  // pop the empty region
  op_result = op_result || parser.parseOptionalRBrace();
  result.addAttribute(getListSizesAttrStrName(),
                      builder.getI32ArrayAttr(list_sizes));

  return failure(op_result);
}

void EqualizePatternOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  auto num_regions = op.getRegions().size();

  p << " {";
  p.increaseIndent();
  for (std::size_t i = 0; i < num_regions; i++) {
    p.printNewline();
    p << '(' << op.getRegionArg(i) << " -> ";
    p.printOperands(op.getRegionUnitList(i));
    p << ')';
    p.printRegion(op.getRegion(i), false, true);
  }
  p.decreaseIndent();
  p.printNewline();
  p << '}';
}

LogicalResult EqualizePatternOp::verify() {
  auto& op = *this;
  assert(op.getRegions().size() >= 1);
  assert(op.getRegions().size() == op.getListSizes().size());

  // no duplication in units
  const auto size = op.getUnits().size();
  for (std::size_t i = 0; i < size - 1; i++) {
    for (std::size_t j = i + 1; j < size; j++) {
      if (op.getUnits()[i] == op.getUnits()[j]) {
        op.emitError("duplicated unit SSAs found!");
        return failure();
      }
    }
  }

  // verify units of unit_lists are either get_unit or create_group
  for (std::size_t idx = 0; idx < op.getRegions().size(); idx++) {
    auto units = op.getRegionUnitList(idx);
    if (units.size() == 1) {
      auto def_op = units.front().getDefiningOp();
      if (!isa<dataflow::GetUnitOp, dataflow::CreateGroupOp>(def_op)) {
        op.emitError(
            "region's unit_list must be programUnitOp or createGroupOp type!");
        return failure();
      }
    } else {
      for (auto unit : units) {
        if (!isa<dataflow::GetUnitOp>(unit.getDefiningOp())) {
          op.emitError(
              "when unit_list size > 1, all units must be programUnitOp!");
          return failure();
        }
      }
    }
  }

  // all regions should have the same number of operations
  const auto region_size = op.getRegions().size();
  assert(region_size > 1);
  const auto operation_size = op.getRegion(0).front().getOperations().size();
  for (std::size_t idx = 1; idx < region_size; idx++) {
    if (op.getRegion(idx).front().getOperations().size() != operation_size) {
      op.emitError("regions have different operation size.");
      return failure();
    }
  }
  // all regions should contain the same sequence of operations.
  SmallVector<OperationName> op_names;
  for (auto& each_op : op.getRegion(0).front().getOperations()) {
    op_names.push_back(each_op.getName());
  }
  for (std::size_t region_idx = 1; region_idx < region_size; region_idx++) {
    auto& region = op.getRegion(region_idx);
    int op_idx = 0;
    for (auto& each_op : region.front().getOperations()) {
      if (each_op.getName() != op_names[op_idx]) {
        op.emitError("Operation sequences different.");
        return failure();
      }
      op_idx++;
    }
  }
  return success();
}

ValueRange EqualizePatternOp::getRegionUnitList(int pos) {
  int drop = 0;
  int keep =
      mlir::cast<IntegerAttr>(this->getListSizes().getValue()[pos]).getInt();
  for (auto idx = 0; idx < pos; idx++) {
    drop +=
        mlir::cast<IntegerAttr>(this->getListSizes().getValue()[idx]).getInt();
  }
  return this->getUnits().slice(drop, keep);
}

std::optional<ValueRange> EqualizePatternOp::getRegionUnitList(Value arg) {
  int size = this->getListSizes().size();
  for (int idx = 0; idx < size; idx++) {
    if (arg == getRegionArg(idx)) {
      return getRegionUnitList(idx);
    }
  }
  return std::nullopt;
}

std::optional<Region*> EqualizePatternOp::getSpecificRegion(Value arg) {
  int size = this->getListSizes().size();
  for (int idx = 0; idx < size; idx++) {
    if (arg == getRegionArg(idx)) {
      return &getRegion(idx);
    }
  }
  return std::nullopt;
}

Region* EqualizePatternOp::getRegionFromUnit(Value unit) {
  int idx_in_unit_list = 0;
  for (auto get_unit : this->getUnits()) {
    if (get_unit == unit) break;
    ++idx_in_unit_list;
  }

  int num_units_visited = 0;
  auto list_sizes = this->getListSizes();
  for (std::size_t i = 0; i < list_sizes.size(); ++i) {
    num_units_visited += mlir::cast<IntegerAttr>(list_sizes[i]).getInt();
    if (num_units_visited > idx_in_unit_list) return &this->getRegion(i);
  }
  return nullptr;
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
    if (parser.parseOptionalComma().failed()) break;
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

DenseMap<Value, Value> DefImmutableMappingOp::toMap() {
  DenseMap<Value, Value> result;
  for (auto [key, value] : llvm::zip(getKeys(), getValues())) {
    result[key] = value;
  }
  return result;
}

void DefImmutableMappingOp::getValuesFromKeys(
    ValueRange keys, SmallVectorImpl<std::optional<Value>>& result) {
  const auto map = toMap();
  for (auto key : keys) {
    if (auto value = map.lookup(key); value) {
      result.emplace_back(value);
    } else {
      result.emplace_back(std::nullopt);
    }
  }
}

void DefImmutableMappingOp::getNonNullValuesFromKeys(
    ValueRange keys, SmallVectorImpl<Value>& result) {
  const auto map = toMap();
  for (auto key : keys) {
    if (const auto value = map.lookup(key); value) {
      result.push_back(value);
    }
  }
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
