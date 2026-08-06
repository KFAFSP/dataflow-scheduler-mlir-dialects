//===-- DataflowOps.cpp ------------------------------------------*- c++ -*-==//
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

// clang-format off
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
// clang-format on

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IntegerSet.h>

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.cpp.inc"

//===----------------------------------------------------------------------===//
// GetUnitOp
//===----------------------------------------------------------------------===//

LogicalResult GetUnitOp::verify() {
  if (getNumResults() < 1) {
    return emitOpError("expected at least 1 result");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ProgramUnitOp
//===----------------------------------------------------------------------===//

void ProgramUnitOp::build(OpBuilder& builder, OperationState& result,
                          ValueRange unitIds, StringAttr precision_attr,
                          ProgramUnitOp::BodyBuilderFn bodyBuilder) {
  auto& props = result.getOrAddProperties<Properties>();

  result.addOperands(unitIds);

  if (!precision_attr.getValue().str().empty()) {
    props.precision = precision_attr;
  }

  // Add a body region with block arguments as unwrapped async value operands.
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block& bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(unitIds[0].getType(), result.location);

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (!bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    dataflow::ReturnOp::create(builder, result.location, TypeRange());
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    // int tmp = bodyBlock.getNumArguments();
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

void ProgramUnitOp::build(OpBuilder& builder, OperationState& result,
                          ValueRange unitIds,
                          ProgramUnitOp::BodyBuilderFn bodyBuilder) {
  build(builder, result, unitIds,
        StringAttr::get(builder.getContext(), StringRef("")), bodyBuilder);
}

ParseResult ProgramUnitOp::parse(OpAsmParser& parser, OperationState& result) {
  ::llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> unitsOperands;
  ::llvm::SMLoc unitsOperandsLoc;
  (void)unitsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<Region>, 2> fullRegions;
  OpAsmParser::Argument region_arg;
  region_arg.type = parser.getBuilder().getIndexType();

  unitsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalKeyword("iter_arg").succeeded()) {
    if (parser.parseColon()) return failure();
    if (parser.parseArgument(region_arg)) return failure();
    if (parser.parseArrow()) return failure();
    if (parser.parseOperandList(unitsOperands, AsmParser::Delimiter::Paren))
      return failure();
  } else {
    if (parser.parseOperandList(unitsOperands)) return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  if (parser.parseColon()) return failure();

  {
    std::unique_ptr<Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region, region_arg);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult)) return failure();
      fullRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<Region>();
        if (parser.parseRegion(*region)) return failure();
        fullRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto& region : fullRegions)
    ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegions(fullRegions);
  Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(unitsOperands, odsBuildableType0, unitsOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void ProgramUnitOp::print(OpAsmPrinter& _odsPrinter) {
  _odsPrinter << ' ';
  auto arguments = getRegion().getArguments();
  if (arguments.size() > 0) {
    _odsPrinter << "iter_arg : ";
    _odsPrinter << arguments;
    _odsPrinter << " -> (";
  }
  _odsPrinter << getUnits();
  if (arguments.size() > 0) {
    _odsPrinter << ')';
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  llvm::interleaveComma(
      getOperation()->getRegions(), _odsPrinter, [&](Region& region) {
        {
          bool printTerminator = true;
          if (auto* term =
                  region.empty() ? nullptr : region.begin()->getTerminator()) {
            printTerminator = !term->getAttrDictionary().empty() ||
                              term->getNumOperands() != 0 ||
                              term->getNumResults() != 0;
          }
          _odsPrinter.printRegion(region, /*printEntryBlockArgs=*/false,
                                  /*printBlockTerminators=*/printTerminator);
        }
      });
}

//===----------------------------------------------------------------------===//
// GetPagedLogicalMemoryViewOp
//===----------------------------------------------------------------------===//

ParseResult GetPagedLogicalMemoryViewOp::parse(OpAsmParser& parser,
                                               OperationState& result) {
  auto& props = result.getOrAddProperties<Properties>();

  // Parse the operands.
  OpAsmParser::UnresolvedOperand unit, start_addr;
  auto op_result = parser.parseOperand(unit) || parser.parseComma() ||
                   parser.parseOperand(start_addr);

  // Parse the attributes.
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes);

  // Parse the segments. We anticipate each segment/page be named "page<#>"
  // where # starts at 0 and increases by 1 for each additional page. There
  // should be at least one page.
  op_result = op_result || parser.parseLBrace() ||
              parser.parseKeyword("segments") || parser.parseEqual();

  SmallVector<Attribute, 4> idx_sets;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> page_start_addrs;
  int page_num = 0;
  std::string page_keyword = "page" + std::to_string(page_num);
  while (parser.parseOptionalKeyword(page_keyword).succeeded()) {
    IntegerSetAttr idx_set;
    OpAsmParser::UnresolvedOperand page_start_addr;
    op_result = op_result || parser.parseColon() || parser.parseLParen() ||
                parser.parseKeyword("idx_set") || parser.parseEqual() ||
                parser.parseAttribute(idx_set) || parser.parseComma() ||
                parser.parseKeyword("start_addr") || parser.parseEqual() ||
                parser.parseOperand(page_start_addr) || parser.parseRParen();
    std::ignore = parser.parseOptionalComma();
    idx_sets.push_back(idx_set);
    page_start_addrs.push_back(page_start_addr);
    ++page_num;
    page_keyword = "page" + std::to_string(page_num);
  }
  op_result = op_result || parser.parseRBrace();

  auto& builder = parser.getBuilder();
  props.idx_sets = builder.getArrayAttr(idx_sets);

  // Parse the types. The operation only displays the type of the unit and
  // start_address operands, and the type of the result of the operation.
  auto index_type = builder.getIndexType();
  Type result_type;
  op_result = op_result || parser.parseColonType(index_type) ||
              parser.parseComma() || parser.parseType(index_type) ||
              parser.parseComma() || parser.parseType(result_type);

  // Parse result type.
  result.types.push_back(result_type);

  op_result =
      op_result || parser.resolveOperand(unit, index_type, result.operands) ||
      parser.resolveOperand(start_addr, index_type, result.operands) ||
      parser.resolveOperands(page_start_addrs, index_type, result.operands);

  return failure(op_result);
}

void GetPagedLogicalMemoryViewOp::print(::mlir::OpAsmPrinter& p) {
  auto& op = *this;

  // Print the unit, start address, and attributes except for the idx_sets.
  p << " " << op.getUnit() << ", " << op.getStartAddr();
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs*/ {op.getIdxSetsStrName()});

  // Print the segments (the idx_set + page_start_address pairs).
  p.printNewline();
  p << "  {segments = ";
  int page_num = 0;
  assert((op.getIdxSets().size() == op.getPageStartAddrs().size()) &&
         "idx_sets attribute must have same number of entries as "
         "page_start_addrs operands");
  for (auto&& [idx_set, page_start_addr] :
       llvm::zip(op.getIdxSets(), op.getPageStartAddrs())) {
    if (page_num > 0) p << ",";
    p.printNewline();
    p << "    page" << page_num << ": (idx_set = " << idx_set << ", "
      << "start_addr = " << page_start_addr << ")";
    ++page_num;
  }
  p << "} : ";
  p.printType(op.getUnit().getType());
  p << ", ";
  p.printType(op.getStartAddr().getType());
  p << ", ";
  p.printType(op.getResult().getType());
}

LogicalResult GetPagedLogicalMemoryViewOp::verify() {
  auto& op = *this;

  auto idx_sets = op.getIdxSets();
  auto page_start_addrs = op.getPageStartAddrs();
  if (idx_sets.size() != page_start_addrs.size())
    return op->emitOpError(
        "there should be a start address and idx_set for every page");

  for (std::size_t page_idx = 0; page_idx < idx_sets.size(); ++page_idx) {
    // The start addresses for the pages need to be constant operations.
    auto page_start_addr_op = page_start_addrs[page_idx].getDefiningOp();
    if (!page_start_addr_op || !isa<arith::ConstantOp>(page_start_addr_op))
      return op->emitOpError(
          "page start addresses should be arith::ConstantOp");

    // The page sets need to be hyper rectangular.
    auto idx_set = idx_sets[page_idx];
    auto page_set = cast<IntegerSetAttr>(idx_set);
    FlatLinearValueConstraints page_set_flat(page_set.getValue());
    if (!page_set_flat.isHyperRectangular(0, page_set_flat.getNumCols() - 1))
      return op->emitOpError("idx_set should be hyper rectangular");
  }
  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// DataflowDialect
//===----------------------------------------------------------------------===//

void DataflowDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.cpp.inc"
      >();
}
