//===-- Agen.cpp -------------------------------------------------*- c++ -*-==//
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
// This file implements the agen dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/Agen.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>

#include <cassert>
#include <memory>
#include <regex>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

using namespace mlir;
using namespace mlir::agen;

//===----------------------------------------------------------------------===//
// Agen Enums
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/AgenEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom Parsers and Printers
//===----------------------------------------------------------------------===//

namespace {

auto parseAffineMapOfSSAIds(
    OpAsmParser& parser, AffineMapAttr& map,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) -> ParseResult {
  // MLIR's parseAffineMapOfSSAIds always builds a NamedAttribute from attrName
  // before returning, and NamedAttribute requires a non-empty name. We return
  // the map via the `map` out-param and discard the named attribute, but the
  // name must still be valid, so pass a placeholder rather than "".
  NamedAttrList ignored;
  return parser.parseAffineMapOfSSAIds(operands, map, "map", ignored,
                                       AsmParser::Delimiter::Square);
}

void printAffineMapOfSSAIds(OpAsmPrinter& printer, Operation* /*op*/,
                            AffineMapAttr map, ValueRange operands) {
  printer << "[";
  printer.printAffineMapOfSSAIds(map, operands);
  printer << "]";
}

}  // namespace

//===----------------------------------------------------------------------===//
// AgenDialect
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Agen/AgenDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/Agen/AgenAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Agen/Agen.cpp.inc"

void AgenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/Agen/Agen.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/Agen/AgenAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// VectorLoadOp
//===----------------------------------------------------------------------===//

auto VectorLoadOp::verify() -> LogicalResult {
  if (getAffineMap().getNumInputs() != getMapOperands().size()) {
    return emitOpError("expected ")
           << getAffineMap().getNumInputs() << " map operands, but got "
           << getMapOperands().size();
  }

  return success();
}

SmallVector<Operation*> VectorLoadOp::getUseChain() {
  auto& op = *this;
  Operation* curr_op = op;
  SmallVector<Operation*> use_chain;
  while (true) {
    use_chain.push_back(curr_op);
    // Only the last operation in the chain should have no results
    // (dataflow.send, for example).
    if (curr_op->getNumResults() == 0) break;
    assert((curr_op->getNumResults() == 1) &&
           "use chain from agen.vector_load op should have operations that "
           "have one result");
    Value res = curr_op->getResult(0);
    assert((res.hasOneUse()) &&
           "ops in the use chain from agen.vector_load op that have results "
           "should have 1 use");
    curr_op = *res.getUsers().begin();
  }
  assert(use_chain.size() >= 2 && use_chain.back()->getNumResults() == 0);

  return use_chain;
}

void VectorLoadOp::cloneUseChainToNewOp(OpBuilder& builder, Operation* new_op) {
  assert(isa<VectorLoadOp>(new_op));
  auto use_chain = getUseChain();

  Value prev_val = nullptr, prev_cloned_val = new_op->getResult(0);
  for (int i = 0, last_op_idx = use_chain.size() - 1; i <= last_op_idx; ++i) {
    // Clone all the ops except the first one. That one is covered by mem_op.
    if (prev_val) {
      auto cloned_op = builder.clone(*use_chain[i]);
      cloned_op->replaceUsesOfWith(prev_val, prev_cloned_val);
      builder.setInsertionPointAfter(cloned_op);
      // If this is the last operation in the chain, it won't have
      // a result.
      if (i != last_op_idx) {
        prev_val = use_chain[i]->getResult(0);
        prev_cloned_val = cloned_op->getResult(0);
      }
    } else {
      prev_val = use_chain[i]->getResult(0);
    }
  }
}

agen::VectorLoadOp VectorLoadOp::cloneWithNewAccessInfo(
    OpBuilder& builder, const Value mem_view, const AffineMap& subscripts_map,
    ValueRange indices) {
  return VectorLoadOp::create(
      builder, getLoc(), getResult().getType(), mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      subscripts_map, indices, getLoadSet(), getLoadOrder(),
      getMulticastInfo());
}

void VectorLoadOp::eraseOpAndUseChain() {
  auto use_chain = getUseChain();
  if (use_chain.empty()) {
    auto& op = *this;
    op->erase();
  } else {
    for (int idx = use_chain.size() - 1; idx >= 0; --idx)
      use_chain[idx]->erase();
  }
}

//===----------------------------------------------------------------------===//
// VectorStoreOp
//===----------------------------------------------------------------------===//

// Use identity map.
void VectorStoreOp::build(OpBuilder& builder, OperationState& state,
                          Value value_to_store, Value mem_ref,
                          /*optional*/ StringAttr dbg_name, ValueRange indices,
                          IntegerSet store_set, AffineMap store_order) {
  const auto memref_type = cast<MemRefType>(mem_ref.getType());
  build(builder, state, value_to_store, mem_ref, dbg_name,
        builder.getMultiDimIdentityMap(memref_type.getRank()), indices,
        store_set, store_order);
}

auto VectorStoreOp::verify() -> LogicalResult {
  if (getAffineMap().getNumInputs() != getMapOperands().size()) {
    return emitOpError("expected ")
           << getAffineMap().getNumInputs() << " map operands, but got "
           << getMapOperands().size();
  }

  return success();
}

SmallVector<Operation*> VectorStoreOp::getUseChain() {
  auto& op = *this;
  SmallVector<Operation*> use_chain;
  use_chain.push_back(op);

  // Store operations have use chains that lead to them. These can be:
  //   receive + store
  //   constant_bitstream + shuffle + store
  //   vector_load + store
  auto input_op = getValueToStore().getDefiningOp();
  use_chain.push_back(input_op);
  if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(input_op))
    use_chain.push_back(shuffle_op.getInput().getDefiningOp());

  return use_chain;
}

void VectorStoreOp::cloneUseChainToNewOp(OpBuilder& builder,
                                         Operation* new_op) {
  assert(isa<VectorStoreOp>(new_op));
  auto use_chain = getUseChain();

  // The first operation in the use chain is the VectorStoreOp.
  // The use chain is stored in reverse order.
  Value prev_val = nullptr, prev_cloned_val = nullptr;
  for (int i = use_chain.size() - 1; i > 0; --i) {
    auto cloned_op = builder.clone(*use_chain[i]);
    if (prev_val && prev_cloned_val)
      cloned_op->replaceUsesOfWith(prev_val, prev_cloned_val);
    builder.setInsertionPointAfter(cloned_op);
    prev_val = use_chain[i]->getResult(0);
    prev_cloned_val = cloned_op->getResult(0);
  }
  new_op->replaceUsesOfWith(prev_val, prev_cloned_val);
}

VectorStoreOp VectorStoreOp::cloneWithNewAccessInfo(
    OpBuilder& builder, Value mem_view, const AffineMap& subscripts_map,
    ValueRange indices) {
  return VectorStoreOp::create(
      builder, getLoc(), getValueToStore(), mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      subscripts_map, indices, getStoreSet(), getStoreOrder());
}

void VectorStoreOp::eraseOpAndUseChain() {
  auto use_chain = getUseChain();
  if (use_chain.empty()) {
    auto& op = *this;
    op->erase();
  } else {
    for (auto& o : use_chain) o->erase();
  }
}

//===----------------------------------------------------------------------===//
// CompositeLoadAndStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeLoadAndStoreOp::parse(OpAsmParser& parser,
                                           OperationState& result) {
  auto& props = result.getOrAddProperties<Properties>();

  OpAsmParser::UnresolvedOperand src_mem_ref;
  SmallVector<OpAsmParser::UnresolvedOperand> src_operands;
  if (parser.parseKeyword("src") || parser.parseColon() ||
      parser.parseOperand(src_mem_ref) ||
      parseAffineMapOfSSAIds(parser, props.src_map, src_operands)) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand dst_mem_ref;
  SmallVector<OpAsmParser::UnresolvedOperand> dst_operands;
  if (parser.parseKeyword("dst") || parser.parseColon() ||
      parser.parseOperand(dst_mem_ref) ||
      parseAffineMapOfSSAIds(parser, props.dst_map, dst_operands)) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> time_symbols;
  if (parser.parseKeyword("time_symbols") ||
      parser.parseOperandList(time_symbols, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }

  OpAsmParser::Argument load_induction_var;
  if (parser.parseComma() || parser.parseKeyword("load_iv") ||
      parser.parseLParen() || parser.parseArgument(load_induction_var, true) ||
      parser.parseRParen()) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand multicast_info;
  if (!parser.parseOptionalComma() &&
      (parser.parseKeyword("multicast_info") || parser.parseEqual() ||
       parser.parseOperand(multicast_info))) {
    return failure();
  }

  // FIXME: Can't have optional attr dict before region.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.parseRegion(*result.addRegion(), {load_induction_var})) {
    return failure();
  }

  Type src_mem_ref_type;
  Type dst_mem_ref_type;
  if (parser.parseColon() || parser.parseType(src_mem_ref_type) ||
      parser.parseComma() || parser.parseType(dst_mem_ref_type)) {
    return failure();
  }

  const auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(src_mem_ref, src_mem_ref_type, result.operands) ||
      parser.resolveOperand(dst_mem_ref, dst_mem_ref_type, result.operands) ||
      parser.resolveOperands(src_operands, index_type, result.operands) ||
      parser.resolveOperands(dst_operands, index_type, result.operands) ||
      parser.resolveOperands(time_symbols, index_type, result.operands)) {
    return failure();
  }
  if (multicast_info.location.isValid() &&
      parser.resolveOperand(multicast_info, index_type, result.operands)) {
    return failure();
  }

  props.operandSegmentSizes = {1,
                               1,
                               static_cast<int32_t>(src_operands.size()),
                               static_cast<int32_t>(dst_operands.size()),
                               static_cast<int32_t>(time_symbols.size()),
                               multicast_info.location.isValid() ? 1 : 0};

  return success();
}

void CompositeLoadAndStoreOp::print(OpAsmPrinter& p) {
  p << " src:" << getSrcMemRef();
  printAffineMapOfSSAIds(p, *this, getSrcMapAttr(), getSrcMapOperands());
  p << " dst:" << getDstMemRef();
  printAffineMapOfSSAIds(p, *this, getDstMapAttr(), getDstMapOperands());

  p.printNewline();
  p << " time_symbols(";
  p.printOperands(getTimeSymbols());
  p << "), load_iv(";
  p << getLoadInductionVar() << ":" << getLoadInductionVar().getType();
  p << ')';

  if (const auto multicast_info = getMulticastInfo(); multicast_info) {
    p << ", multicast_info = " << multicast_info;
  }

  p.printNewline();
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSrcMapAttrName(), getDstMapAttrName(),
                       getOperandSegmentSizesAttrName()});

  p.printNewline();
  p.printRegion(getBodyRegion(), false, true);
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType();
}

void CompositeLoadAndStoreOp::build(
    OpBuilder& builder, OperationState& state, Value src_mem_ref,
    Value dst_mem_ref, StringAttr dbg_name, AffineMapAttr src_map,
    ValueRange src_map_operands, AffineMapAttr dst_map,
    ValueRange dst_map_operands, IntegerSetAttr load_set,
    AffineMapAttr load_order, IntegerSetAttr store_set,
    AffineMapAttr store_order, ValueRange time_symbols, IntegerSetAttr time_set,
    AffineMapAttr time_order, AffineMapAttr load_time_addr_map,
    AffineMapAttr store_time_addr_map, VectorType type,
    AgenRoutingDirectionAttr dir, Value multicast_info,
    function_ref<void(OpBuilder&, Location, Value)> body_builder) {
  state.addOperands({src_mem_ref, dst_mem_ref});
  state.addOperands(src_map_operands);
  state.addOperands(dst_map_operands);
  state.addOperands(time_symbols);
  if (multicast_info) {
    state.operands.push_back(multicast_info);
  }

  auto& props = state.getOrAddProperties<Properties>();

  props.src_map = src_map;
  props.dst_map = dst_map;
  props.load_set = load_set;
  props.load_order = load_order;
  props.store_set = store_set;
  props.store_order = store_order;
  props.time_set = time_set;
  props.time_order = time_order;
  props.load_time_addr_map = load_time_addr_map;
  props.store_time_addr_map = store_time_addr_map;
  props.dbgName = dbg_name;
  props.dir = dir;

  props.operandSegmentSizes = {1,
                               1,
                               static_cast<int32_t>(src_map_operands.size()),
                               static_cast<int32_t>(dst_map_operands.size()),
                               static_cast<int32_t>(time_symbols.size()),
                               multicast_info ? 1 : 0};

  auto* body = &state.addRegion()->emplaceBlock();
  const auto load_induction_var = body->addArgument(type, state.location);
  ensureTerminator(*state.regions.front(), builder, state.location);

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    body_builder(builder, state.location, load_induction_var);
  }
}

void CompositeLoadAndStoreOp::build(
    OpBuilder& builder, OperationState& state, Value src_mem_ref,
    Value dst_mem_ref, StringAttr dbg_name, AffineMap src_map,
    ValueRange src_operands, AffineMap dst_map, ValueRange dst_operands,
    IntegerSet load_set, AffineMap load_order, IntegerSet store_set,
    AffineMap store_order, ValueRange time_symbols, IntegerSet time_set,
    AffineMap time_order, AffineMap load_time_addr_map,
    AffineMap store_time_addr_map, VectorType type,
    AgenRoutingDirectionAttr dir, Value multicast_info,
    function_ref<void(OpBuilder&, Location, Value)> body_builder) {
  build(builder, state, src_mem_ref, dst_mem_ref, dbg_name,
        AffineMapAttr::get(src_map), src_operands, AffineMapAttr::get(dst_map),
        dst_operands, IntegerSetAttr::get(load_set),
        AffineMapAttr::get(load_order), IntegerSetAttr::get(store_set),
        AffineMapAttr::get(store_order), time_symbols,
        IntegerSetAttr::get(time_set), AffineMapAttr::get(time_order),
        AffineMapAttr::get(load_time_addr_map),
        AffineMapAttr::get(store_time_addr_map), type, dir, multicast_info,
        body_builder);
}

auto CompositeLoadAndStoreOp::verify() -> LogicalResult {
  if (getLoadSet().getValue().getNumDims() < getSrcMap().getNumResults()) {
    return emitOpError() << "load set and src array dimensions should match";
  }

  if (getLoadSet().getValue().getNumDims() != getLoadOrder().getNumDims()) {
    return emitOpError() << "load set and order dimensions should match";
  }

  if (getStoreSet().getValue().getNumDims() < getDstMap().getNumResults()) {
    return emitOpError() << "store set and dst array dimensions should match";
  }

  if (getStoreSet().getValue().getNumDims() != getStoreOrder().getNumDims()) {
    return emitOpError() << "store set and order dimensions should match";
  }

  return success();
}

auto CompositeLoadAndStoreOp::getAffineMapAttrForMemRef(Value memref)
    -> NamedAttribute {
  if (memref == getSrcMemRef()) {
    return {getSrcMapAttrName(), getSrcMapAttr()};
  }

  if (memref == getDstMemRef()) {
    return {getDstMapAttrName(), getDstMapAttr()};
  }

  llvm_unreachable("invalid operand value");
}

CompositeLoadAndStoreOp CompositeLoadAndStoreOp::cloneWithNewAccessInfo(
    OpBuilder& builder, Value src_mem_view, Value dst_mem_view,
    const AffineMap& src_subscripts_map, const AffineMap& dst_subscripts_map,
    ValueRange src_indices, ValueRange dst_indices,
    const IntegerSet& time_set) {
  auto result = CompositeLoadAndStoreOp::create(
      builder, getLoc(), src_mem_view, dst_mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      src_subscripts_map, src_indices, dst_subscripts_map, dst_indices,
      getLoadSet().getValue(), getLoadOrder(), getStoreSet().getValue(),
      getStoreOrder(), getTimeSymbols(), time_set, getTimeOrder(),
      getLoadTimeAddrMap(), getStoreTimeAddrMap(),
      getLoadInductionVar().getType(),
      mlir::agen::AgenRoutingDirectionAttr::get(
          getContext(), AgenRoutingDirection::PseudoRandom),
      getMulticastInfo());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = result.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&result.getRegion().front());
  IRMapping ir_map;
  for (auto& o : getRegion().getOps()) (void)builder.clone(*&o, ir_map);
  builder.restoreInsertionPoint(insert_pt);

  // Replace any uses of the old load_iv with the new one.
  auto orig_load_iv = getLoadInductionVar();
  auto new_load_iv = result.getLoadInductionVar();
  for (auto& o : result.getRegion().getOps())
    o.replaceUsesOfWith(orig_load_iv, new_load_iv);

  return result;
}

//===----------------------------------------------------------------------===//
// CompositeLoadOp
//===----------------------------------------------------------------------===//

ParseResult CompositeLoadOp::parse(OpAsmParser& parser,
                                   OperationState& result) {
  auto& props = result.getOrAddProperties<Properties>();

  OpAsmParser::UnresolvedOperand mem_ref;
  SmallVector<OpAsmParser::UnresolvedOperand> map_operands;
  if (parser.parseOperand(mem_ref) ||
      parseAffineMapOfSSAIds(parser, props.affine_map, map_operands)) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> time_symbols;
  if (parser.parseKeyword("time_symbols") ||
      parser.parseOperandList(time_symbols, AsmParser::Delimiter::Paren)) {
    return failure();
  }

  OpAsmParser::Argument induction_var;
  if (parser.parseLParen() || parser.parseArgument(induction_var, true) ||
      parser.parseRParen()) {
    return failure();
  }

  // FIXME: Can't have optional attr dict before region.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.parseRegion(*result.addRegion(), {induction_var})) {
    return failure();
  }

  Type mem_ref_type;
  if (parser.parseColonType(mem_ref_type)) {
    return failure();
  }

  const auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(mem_ref, mem_ref_type, result.operands) ||
      parser.resolveOperands(map_operands, index_type, result.operands) ||
      parser.resolveOperands(time_symbols, index_type, result.operands)) {
    return failure();
  }

  props.operandSegmentSizes = {1, static_cast<int32_t>(map_operands.size()),
                               static_cast<int32_t>(time_symbols.size())};

  return success();
}

auto CompositeLoadOp::verify() -> LogicalResult {
  const auto affine_map = getAffineMap();

  // TODO: enhance this with finding constant values and make sure they match.
  const auto load_set = getLoadSet().getValue();
  if (load_set.getNumDims() < affine_map.getNumResults()) {
    return emitOpError() << "load set and Array dimensions should match";
  }

  const auto load_order = getLoadOrder();
  if (load_set.getNumDims() != load_order.getNumDims()) {
    return emitOpError() << "load set and order dimensions should match";
  }

  return success();
}

void CompositeLoadOp::print(OpAsmPrinter& p) {
  p << ' ' << getMemRef();
  printAffineMapOfSSAIds(p, *this, getAffineMapAttr(), getMapOperands());
  p.printNewline();
  p << " time_symbols(";
  p << getTimeSymbols();
  p << ')';
  p << '(' << getLoadInductionVar() << ':' << getLoadInductionVar().getType()
    << ')';
  p.printNewline();
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getMapAttrStrName(), getOperandSegmentSizesAttrName()});
  p.printNewline();
  p.printRegion(getRegion(), false, true);
  p << " : " << getMemRef().getType();
}

void CompositeLoadOp::build(OpBuilder& builder, OperationState& result,
                            Value memref, StringAttr dbg_name,
                            AffineMap affine_map, ValueRange map_operands,
                            VectorType load_type, IntegerSet load_set,
                            AffineMap load_order, ValueRange time_symbols,
                            IntegerSet time_set, AffineMap time_order,
                            AffineMap time_addr_map) {
  auto& props = result.getOrAddProperties<Properties>();

  // Checks
  assert((load_set.getNumDims() == load_order.getNumDims()) &&
         "The number of inputs in load set should match with number of "
         "inputs in load order");
  assert((load_order.getNumDims() == load_order.getNumResults()) &&
         "The number of inputs and results in load order should match "
         "with each other");
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == time_addr_map.getNumDims()) &&
         "The number of inputs in time set and time address map should "
         "match with each other");

  result.addOperands(memref);
  result.addOperands(map_operands);

  props.dbgName = dbg_name;
  props.load_set = IntegerSetAttr::get(load_set);
  props.affine_map = AffineMapAttr::get(affine_map);
  props.load_order = AffineMapAttr::get(load_order);
  props.time_set = IntegerSetAttr::get(time_set);
  props.time_order = AffineMapAttr::get(time_order);
  props.time_addr_map = AffineMapAttr::get(time_addr_map);
  props.operandSegmentSizes = {1, static_cast<int32_t>(map_operands.size()),
                               static_cast<int32_t>(time_symbols.size())};

  // Add a body region with block arguments
  auto& body = result.addRegion()->emplaceBlock();
  // FIXME: which getLoc() should be used here?
  body.addArgument(load_type, memref.getLoc());
  CompositeLoadOp::ensureTerminator(*body.getParent(), builder,
                                    result.location);
}

CompositeLoadOp CompositeLoadOp::cloneWithNewAccessInfo(
    OpBuilder& builder, Value mem_view, const AffineMap& subscripts_map,
    ValueRange indices, const IntegerSet& time_set) {
  auto result = CompositeLoadOp::create(
      builder, getLoc(), mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      subscripts_map, indices, getLoadInductionVar().getType(),
      getLoadSet().getValue(), getLoadOrder(), getTimeSymbols(), time_set,
      getTimeOrder(), getTimeAddrMap());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = result.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(result.getBody());
    IRMapping ir_map;
    for (auto& o : getRegion().getOps()) {
      std::ignore = builder.clone(o, ir_map);
    }
  }

  // Replace any uses of the old load_iv with the new one.
  auto orig_load_iv = getLoadInductionVar();
  auto new_load_iv = result.getLoadInductionVar();
  for (auto& o : result.getRegion().getOps())
    o.replaceUsesOfWith(orig_load_iv, new_load_iv);

  return result;
}

//===----------------------------------------------------------------------===//
// CompositeStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeStoreOp::parse(OpAsmParser& parser,
                                    OperationState& result) {
  auto& props = result.getOrAddProperties<Properties>();

  OpAsmParser::UnresolvedOperand mem_ref;
  SmallVector<OpAsmParser::UnresolvedOperand> map_operands;
  if (parser.parseOperand(mem_ref) ||
      parseAffineMapOfSSAIds(parser, props.affine_map, map_operands)) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand input_vector;
  if (succeeded(parser.parseOptionalKeyword("input_vector")) &&
      parser.parseEqual() && parser.parseOperand(input_vector)) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> time_symbols;
  if (parser.parseKeyword("time_symbols") ||
      parser.parseOperandList(time_symbols, AsmParser::Delimiter::Paren)) {
    return failure();
  }

  // FIXME: Can't have optional attr dict before region.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // FIXME: We never parsed any region arguments.
  std::unique_ptr<Region> region;
  if (const auto maybe = parser.parseOptionalRegion(region, {});
      maybe.has_value() && maybe.value()) {
    return failure();
  }

  if (region) {
    if (input_vector.location.isValid()) {
      return parser.emitError(parser.getNameLoc())
             << "for composite_store, input_vector and region can't co-exist";
    }
  } else {
    region = std::make_unique<Region>();
  }

  CompositeStoreOp::ensureTerminator(*region, parser.getBuilder(),
                                     result.location);
  result.addRegion(std::move(region));

  Type mem_ref_type;
  if (parser.parseColonType(mem_ref_type)) {
    return failure();
  }

  Type input_vector_type;
  if (input_vector.location.isValid()) {
    if (parser.parseComma() || parser.parseType(input_vector_type)) {
      return failure();
    }
  }

  const auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(mem_ref, mem_ref_type, result.operands) ||
      parser.resolveOperands(map_operands, index_type, result.operands)) {
    return failure();
  }
  if (input_vector.location.isValid() &&
      parser.resolveOperand(input_vector, input_vector_type, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(time_symbols, index_type, result.operands)) {
    return failure();
  }

  props.operandSegmentSizes = {
      1,
      static_cast<int32_t>(map_operands.size()),
      input_vector.location.isValid() ? 1 : 0,
      static_cast<int32_t>(time_symbols.size()),
  };

  if (input_vector.location.isValid()) {
    // for coalesce store, store_order attr is not allowed from users. so set
    // store_order to Identity map
    auto num_of_layout_dims = props.affine_map.getValue().getResults().size();
    props.store_order = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
        num_of_layout_dims, parser.getBuilder().getContext()));
  }

  return success();
}

auto CompositeStoreOp::verify() -> LogicalResult {
  const auto has_input_vector = getInputVector() != nullptr;
  const auto store_set = getStoreSet();
  const auto store_order = getStoreOrder();
  if (store_set.has_value() == has_input_vector ||
      store_order.has_value() == has_input_vector) {
    return emitOpError()
           << "either store_set/order or input vector has to be present in "
              "composite store";
  }

  if (!has_input_vector) {
    // TODO: enhance this with finding constant values and make sure they match.
    if (store_set->getValue().getNumDims() < getAffineMap().getNumResults()) {
      return emitOpError() << "store set and Array dimensions should match";
    }

    if (store_set->getValue().getNumDims() != store_order->getNumDims()) {
      return emitOpError() << "store set and order dimensions should match";
    }
  }

  return success();
}

void CompositeStoreOp::print(OpAsmPrinter& p) {
  p << ' ' << getMemRef();
  printAffineMapOfSSAIds(p, *this, getAffineMapAttr(), getMapOperands());
  p.printNewline();
  const auto input_vector = getInputVector();
  if (input_vector != nullptr) {
    p << "input_vector=" << input_vector;
  }
  p << " time_symbols(";
  p.printOperands(getTimeSymbols());
  p << ')';
  p.printNewline();
  if (input_vector != nullptr) {
    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        /*elidedAttrs=*/{getMapAttrStrName(), getOperandSegmentSizesAttrName(),
                         getStoreOrderAttrName(), getStoreSetAttrName()});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{getMapAttrStrName(),
                                             getOperandSegmentSizesAttrName()});
  }
  p.printNewline();
  if (input_vector == nullptr) {
    p.printRegion(getRegion(), false, true);
  }
  p << " : " << getMemRef().getType();
  if (input_vector != nullptr) {
    p << " , " << input_vector.getType();
  }
}

void CompositeStoreOp::build(OpBuilder& builder, OperationState& result,
                             Value memref,
                             /*optional*/ StringAttr dbg_name, AffineMap map,
                             ValueRange map_operands,
                             /*optional*/ IntegerSet store_set,
                             /*optional*/ AffineMap store_order,
                             ValueRange time_symbols, IntegerSet time_set,
                             AffineMap time_order, AffineMap time_addr_map) {
  auto& props = result.getOrAddProperties<Properties>();

  assert((store_set.getNumDims() == store_order.getNumDims()) &&
         "The number of inputs in store set should match with number of "
         "inputs in store order");
  assert((store_order.getNumDims() == store_order.getNumResults()) &&
         "The number of inputs and results in store order should match "
         "with each other");
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == time_addr_map.getNumDims()) &&
         "The number of inputs in time set and time address map should "
         "match with each other");

  result.addOperands(memref);
  result.addOperands(map_operands);
  result.addOperands(time_symbols);

  props.dbgName = dbg_name;
  props.affine_map = AffineMapAttr::get(map);
  props.store_set = IntegerSetAttr::get(store_set);
  props.store_order = AffineMapAttr::get(store_order);
  props.time_set = IntegerSetAttr::get(time_set);
  props.time_order = AffineMapAttr::get(time_order);
  props.time_addr_map = AffineMapAttr::get(time_addr_map);

  props.operandSegmentSizes = {1, static_cast<int32_t>(map_operands.size()), 0,
                               static_cast<int32_t>(time_symbols.size())};

  for (auto operand : result.operands) {
    bool is_vector_type =
        mlir::isa<VectorType, dataflow::CustomVectorType>(operand.getType());
    assert((!is_vector_type) &&
           "Operands of composite_store op can't contain VectorType item for "
           "non-coalesce store");
  }

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeStoreOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void CompositeStoreOp::build(OpBuilder& builder, OperationState& result,
                             Value memref,
                             /*optional*/ StringAttr dbg_name, AffineMap map,
                             ValueRange map_operands, Value input_vector,
                             ValueRange time_symbols, IntegerSet time_set,
                             AffineMap time_order, AffineMap time_addr_map) {
  auto& props = result.getOrAddProperties<Properties>();

  // checks
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == time_addr_map.getNumDims()) &&
         "The number of inputs in time set and time address map should "
         "match with each other");

  result.addOperands(memref);
  result.addOperands(map_operands);
  result.addOperands(input_vector);
  result.addOperands(time_symbols);

  props.dbgName = dbg_name;
  props.affine_map = AffineMapAttr::get(map);
  props.time_set = IntegerSetAttr::get(time_set);
  props.time_order = AffineMapAttr::get(time_order);
  props.time_addr_map = AffineMapAttr::get(time_addr_map);

  props.operandSegmentSizes = {1, static_cast<int32_t>(map_operands.size()), 1,
                               static_cast<int32_t>(time_symbols.size())};

  const auto is_vector =
      isa<VectorType, dataflow::CustomVectorType>(input_vector.getType());
  assert(is_vector && "input vector is required for coalesce store");

  // for coalesce store, store_order attr is not allowed from users. so set
  // store_order to Identity map
  auto num_of_layout_dims = map.getResults().size();
  auto store_order_map = AffineMap::getMultiDimIdentityMap(
      num_of_layout_dims, builder.getContext());
  props.store_order = AffineMapAttr::get(store_order_map);

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeStoreOp::ensureTerminator(*bodyRegion, builder, result.location);
}

CompositeStoreOp CompositeStoreOp::cloneWithNewAccessInfo(
    OpBuilder& builder, Value mem_view, const AffineMap& subscripts_map,
    ValueRange indices, const IntegerSet& time_set) {
  // FIXME: This code does not handle the case when getInputVector() != nullptr.
  //        It didn't do this before either: while it added it to the operands,
  //        it did not select the builder to call based on it. It always calls
  //        the following builder, which handles the no input vector case:
  auto result = CompositeStoreOp::create(
      builder, getLoc(), mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      subscripts_map, indices,
      getStoreSetAttr() ? getStoreSetAttr().getValue() : IntegerSet(nullptr),
      getStoreOrder().value_or(AffineMap(nullptr)), getTimeSymbols(), time_set,
      getTimeOrder(), getTimeAddrMap());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = result.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&result.getRegion().front());
  IRMapping ir_map;
  for (auto& o : getRegion().getOps()) (void)builder.clone(*&o, ir_map);
  builder.restoreInsertionPoint(insert_pt);

  return result;
}

//===----------------------------------------------------------------------===//
// CompositeIndirectLoadAndStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeIndirectLoadAndStoreOp::parse(OpAsmParser& parser,
                                                   OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType ind_src_memref_type, dir_src_memref_type, ind_dst_memref_type,
      dir_dst_memref_type;
  Type inductionVar_type;
  OpAsmParser::UnresolvedOperand ind_src_memref_info, dir_src_memref_info,
      ind_dst_memref_info, dir_dst_memref_info, multicast_info;
  OpAsmParser::Argument inductionVariable;
  AffineMapAttr ind_src_map_attr, dir_src_map_attr, ind_dst_map_attr,
      dir_dst_map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> ind_src_map_operands,
      dir_src_map_operands, ind_dst_map_operands, dir_dst_map_operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> time_symbols_operands;
  // Parse region arguments.
  SmallVector<OpAsmParser::Argument, 1> regionArgs;
  SmallVector<Type, 1> argTypes;
  Region* body = result.addRegion();

  bool has_ind_src = false, has_ind_dst = false, has_multicast = false;

  bool op_result = false;
  if (succeeded(parser.parseOptionalKeyword("indirect_src"))) {
    op_result = op_result || parser.parseColon() ||
                parser.parseOperand(ind_src_memref_info) ||
                parser.parseAffineMapOfSSAIds(
                    ind_src_map_operands, ind_src_map_attr,
                    CompositeIndirectLoadAndStoreOp::getIndirectSrcMapAttrName(
                        result.name),
                    result.attributes);
    has_ind_src = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing indirect src";
  op_result =
      op_result || parser.parseKeyword("direct_src") || parser.parseColon() ||
      parser.parseOperand(dir_src_memref_info) ||
      parser.parseAffineMapOfSSAIds(
          dir_src_map_operands, dir_src_map_attr,
          CompositeIndirectLoadAndStoreOp::getDirectSrcMapAttrName(result.name),
          result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct src";
  if (succeeded(parser.parseOptionalKeyword("indirect_dst"))) {
    op_result = op_result || parser.parseColon() ||
                parser.parseOperand(ind_dst_memref_info) ||
                parser.parseAffineMapOfSSAIds(
                    ind_dst_map_operands, ind_dst_map_attr,
                    CompositeIndirectLoadAndStoreOp::getIndirectDstMapAttrName(
                        result.name),
                    result.attributes);
    has_ind_dst = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing indirect dst";
  op_result =
      op_result || parser.parseKeyword("direct_dst") || parser.parseColon() ||
      parser.parseOperand(dir_dst_memref_info) ||
      parser.parseAffineMapOfSSAIds(
          dir_dst_map_operands, dir_dst_map_attr,
          CompositeIndirectLoadAndStoreOp::getDirectDstMapAttrName(result.name),
          result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct dst";
  if (succeeded(parser.parseOptionalKeyword("multicast_info"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(multicast_info);
    has_multicast = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing multicast_info";
  op_result = op_result || parser.parseKeyword("time_symbols") ||
              parser.parseOperandList(time_symbols_operands,
                                      OpAsmParser::Delimiter::Paren) ||
              parser.parseComma() || parser.parseKeyword("load_iv") ||
              parser.parseLParen() || parser.parseArgument(inductionVariable) ||
              parser.parseColonType(inductionVar_type) || parser.parseRParen();
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing time_symbols and ivs";

  // Induction variable.
  inductionVariable.type = inductionVar_type;
  regionArgs.push_back(inductionVariable);
  argTypes.push_back(inductionVar_type);

  result.attributes.push_back(builder.getNamedAttr(
      CompositeIndirectLoadAndStoreOp::getOperandSegmentSizesAttrName(
          OperationName(CompositeIndirectLoadAndStoreOp::getOperationName(),
                        builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {has_ind_src ? 1 : 0, 1 /* dir_src_memref */, has_ind_dst ? 1 : 0,
           1 /* dir_dst_memref */, (int)ind_src_map_operands.size(),
           (int)dir_src_map_operands.size(), (int)ind_dst_map_operands.size(),
           (int)dir_dst_map_operands.size(), has_multicast ? 1 : 0,
           (int)time_symbols_operands.size()})));
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes) ||
              parser.parseRegion(*body, regionArgs);

  op_result = op_result || parser.parseColon();
  if (!op_result && has_ind_src)
    op_result = op_result || parser.parseType(ind_src_memref_type) ||
                parser.parseComma() ||
                parser.resolveOperand(ind_src_memref_info, ind_src_memref_type,
                                      result.operands);
  op_result = op_result || parser.parseType(dir_src_memref_type) ||
              parser.parseComma() ||
              parser.resolveOperand(dir_src_memref_info, dir_src_memref_type,
                                    result.operands);

  if (!op_result && has_ind_dst)
    op_result = op_result || parser.parseType(ind_dst_memref_type) ||
                parser.parseComma() ||
                parser.resolveOperand(ind_dst_memref_info, ind_dst_memref_type,
                                      result.operands);

  op_result = op_result || parser.parseType(dir_dst_memref_type) ||
              parser.resolveOperand(dir_dst_memref_info, dir_dst_memref_type,
                                    result.operands);

  if (op_result) return failure();

  // Adding map operands to the result operands after all the memref's.
  if (has_ind_src)
    op_result =
        op_result || parser.resolveOperands(ind_src_map_operands, index_type,
                                            result.operands);

  op_result = op_result || parser.resolveOperands(dir_src_map_operands,
                                                  index_type, result.operands);

  if (has_ind_dst)
    op_result =
        op_result || parser.resolveOperands(ind_dst_map_operands, index_type,
                                            result.operands);

  op_result = op_result || parser.resolveOperands(dir_dst_map_operands,
                                                  index_type, result.operands);

  if (has_multicast)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  op_result = op_result || parser.resolveOperands(time_symbols_operands,
                                                  index_type, result.operands);

  std::optional<NamedAttribute> load_set_attr = result.attributes.getNamed(
      CompositeIndirectLoadAndStoreOp::getLoadSetAttrName(result.name));
  if (!load_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "load set is missing in the operation";
  }
  auto load_set =
      mlir::dyn_cast<IntegerSetAttr>(load_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (load_set.getNumDims() < dir_src_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "load set and direct src array dimensions should match";
  }
  std::optional<NamedAttribute> load_order_attr = result.attributes.getNamed(
      CompositeIndirectLoadAndStoreOp::getLoadOrderAttrName(result.name));
  if (load_order_attr.has_value()) {
    auto load_order =
        mlir::dyn_cast<AffineMapAttr>(load_order_attr->getValue()).getValue();
    if (load_set.getNumDims() != load_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "load set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "load order is missing in the operation";
  }

  std::optional<NamedAttribute> store_set_attr = result.attributes.getNamed(
      CompositeIndirectLoadAndStoreOp::getStoreSetAttrName(result.name));
  if (!store_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "store set is missing in the operation";
  }
  auto store_set =
      mlir::dyn_cast<IntegerSetAttr>(store_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (store_set.getNumDims() < dir_dst_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "store set and direct dst array dimensions should match";
  }
  std::optional<NamedAttribute> store_order_attr = result.attributes.getNamed(
      CompositeIndirectLoadAndStoreOp::getStoreOrderAttrName(result.name));
  if (store_order_attr.has_value()) {
    auto store_order =
        mlir::dyn_cast<AffineMapAttr>(store_order_attr->getValue()).getValue();
    if (store_set.getNumDims() != store_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "store set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "store order is missing in the operation";
  }

  return failure(op_result);
}
void CompositeIndirectLoadAndStoreOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  if (hasIndirectSrc()) {
    p << " indirect_src:" << op.getIndirectSrcMemref() << '[';
    if (AffineMapAttr ind_src_map_attr = getIndirectSrcMapAttr();
        ind_src_map_attr)
      p.printAffineMapOfSSAIds(ind_src_map_attr, op.getIndirectSrcMapIndices());
    p << ']';
  }

  p << " direct_src:" << op.getDirectSrcMemref() << '[';
  if (AffineMapAttr dir_src_map_attr = getDirectSrcMapAttr(); dir_src_map_attr)
    p.printAffineMapOfSSAIds(dir_src_map_attr, op.getDirectSrcMapIndices());
  p << ']';

  if (hasIndirectDst()) {
    p << " indirect_dst:" << op.getIndirectDstMemref() << '[';
    if (AffineMapAttr ind_dst_map_attr = getIndirectDstMapAttr();
        ind_dst_map_attr)
      p.printAffineMapOfSSAIds(ind_dst_map_attr, op.getIndirectDstMapIndices());
    p << ']';
  }

  p << " direct_dst:" << op.getDirectDstMemref() << '[';
  if (AffineMapAttr dir_dst_map_attr = getDirectDstMapAttr(); dir_dst_map_attr)
    p.printAffineMapOfSSAIds(dir_dst_map_attr, op.getDirectDstMapIndices());
  p << ']';

  p.printNewline();
  if (hasMulticastInfo()) p << " multicast_info = " << op.getMulticastInfo();

  p << " time_symbols(";
  p.printOperands(op.getTimeSymbols());
  p << "), load_iv(" << op.getLoadInductionVar() << ':'
    << op.getLoadInductionVarType() << ')';

  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{
          op.getIndirectSrcMapAttrName(), op.getDirectSrcMapAttrName(),
          op.getIndirectDstMapAttrName(), op.getDirectDstMapAttrName(),
          op.getOperandSegmentSizesAttrName()});
  p.printNewline();
  p.printRegion(op.getRegion(), false, true);
  p << " : ";
  if (hasIndirectSrc()) p << op.getIndirectSrcMemRefType() << ", ";
  p << op.getDirectSrcMemRefType() << ", ";
  if (hasIndirectDst()) p << op.getIndirectDstMemRefType() << ", ";
  p << op.getDirectDstMemRefType();
}

void CompositeIndirectLoadAndStoreOp::build(
    ::mlir::OpBuilder& builder, ::mlir::OperationState& result,
    Value indirect_src_memref, Value direct_src_memref,
    Value indirect_dst_memref, Value direct_dst_memref,
    /*optional*/ mlir::StringAttr dbgName, mlir::AffineMap indirect_src_map,
    mlir::AffineMap direct_src_map, mlir::AffineMap indirect_dst_map,
    mlir::AffineMap direct_dst_map, ValueRange operands, Type type,
    IntegerSet load_set, mlir::AffineMap load_order, IntegerSet store_set,
    mlir::AffineMap store_order, IntegerSet time_set,
    mlir::AffineMap time_order, mlir::AffineMap load_indirect_time_addr_map,
    mlir::AffineMap load_direct_time_addr_map,
    mlir::AffineMap store_indirect_time_addr_map,
    mlir::AffineMap store_direct_time_addr_map,
    uint32_t num_ind_src_memref_indices, uint32_t num_dir_src_memref_indices,
    uint32_t num_ind_dst_memref_indices, uint32_t num_dir_dst_memref_indices,
    uint32_t num_multicast_info, uint32_t num_time_symbols,
    CompositeIndirectLoadAndStoreOp::BodyBuilderFn bodyBuilder) {
  assert(((num_multicast_info == 0 || num_multicast_info == 1)) &&
         "expected either no multicast operand or a single SSA value");
  assert((direct_src_memref && direct_dst_map) &&
         "direct src/dst memrefs are mandatory");
  assert((num_dir_src_memref_indices > 0 && num_dir_dst_memref_indices > 0) &&
         "indices are mandatory for direct src/dst memrefs");
  assert(((num_ind_src_memref_indices == 0 || indirect_src_memref)) &&
         "when there are ind_src_memref indices there should be a valid "
         "ind_src_memref");
  assert(((num_ind_dst_memref_indices == 0 || indirect_dst_memref)) &&
         "when there are ind_dst_memref indices there should be a valid "
         "ind_dst_memref");

  // Checks
  assert((load_set.getNumDims() == load_order.getNumDims()) &&
         "The number of inputs in load set should match with number of "
         "inputs in load order");
  assert((load_order.getNumDims() == load_order.getNumResults()) &&
         "The number of inputs and results in load order should match "
         "with each other");
  assert((store_set.getNumDims() == store_order.getNumDims()) &&
         "The number of inputs in store set should match with number of "
         "inputs in store order");
  assert((store_order.getNumDims() == store_order.getNumResults()) &&
         "The number of inputs and results in store order should match "
         "with each other");
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == load_direct_time_addr_map.getNumDims()) &&
         "The number of inputs in time set and load direct time address map "
         "should match with each other");
  assert((time_set.getNumDims() == store_direct_time_addr_map.getNumDims()) &&
         "The number of inputs in time set and store time address map should "
         "match with each other");

  if (indirect_src_memref) {
    assert(
        (time_set.getNumDims() == load_indirect_time_addr_map.getNumDims()) &&
        "The number of inputs in time set and load indirect time address map "
        "should match with each other");
    assert(num_ind_src_memref_indices > 0);
    result.addOperands(indirect_src_memref);
  }

  result.addOperands(direct_src_memref);
  if (indirect_dst_memref) {
    assert(
        (time_set.getNumDims() == store_indirect_time_addr_map.getNumDims()) &&
        "The number of inputs in time set and store indirect time address map "
        "should "
        "match with each other");
    assert(num_ind_dst_memref_indices > 0);
    result.addOperands(indirect_dst_memref);
  }
  result.addOperands(direct_dst_memref);

  result.addOperands(operands);

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getIndirectSrcMapAttrName(result.name),
                      AffineMapAttr::get(indirect_src_map));
  result.addAttribute(getDirectSrcMapAttrName(result.name),
                      AffineMapAttr::get(direct_src_map));
  result.addAttribute(getIndirectDstMapAttrName(result.name),
                      AffineMapAttr::get(indirect_dst_map));
  result.addAttribute(getDirectDstMapAttrName(result.name),
                      AffineMapAttr::get(direct_dst_map));

  result.addAttribute(getLoadSetAttrName(result.name),
                      IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderAttrName(result.name),
                      AffineMapAttr::get(load_order));
  result.addAttribute(getStoreOrderAttrName(result.name),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrName(result.name),
                      IntegerSetAttr::get(store_set));
  result.addAttribute(getTimeSetAttrName(result.name),
                      IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderAttrName(result.name),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getLoadIndirectTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(load_indirect_time_addr_map));
  result.addAttribute(getLoadDirectTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(load_direct_time_addr_map));
  result.addAttribute(getStoreIndirectTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(store_indirect_time_addr_map));
  result.addAttribute(getStoreDirectTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(store_direct_time_addr_map));

  result.addAttribute(
      CompositeIndirectLoadAndStoreOp::getOperandSegmentSizesAttrName(
          OperationName(CompositeIndirectLoadAndStoreOp::getOperationName(),
                        builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {indirect_src_memref ? 1 : 0, 1 /* dir_src_memref */,
           indirect_dst_memref ? 1 : 0, 1 /* dir_dst_memref */,
           (int)num_ind_src_memref_indices, (int)num_dir_src_memref_indices,
           (int)num_ind_dst_memref_indices, (int)num_dir_dst_memref_indices,
           (int)num_multicast_info, (int)num_time_symbols}));

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block& bodyBlock = bodyRegion->front();
  // FIXME: which getLoc() should be used here?
  bodyBlock.addArgument(type, direct_src_memref.getLoc());
  CompositeIndirectLoadAndStoreOp::ensureTerminator(*bodyRegion, builder,
                                                    result.location);
}

CompositeIndirectLoadAndStoreOp
CompositeIndirectLoadAndStoreOp::cloneWithNewAccessInfo(
    OpBuilder& builder, const Value& ind_src_mem_view,
    const Value& dir_src_mem_view, const Value& ind_dst_mem_view,
    const Value& dir_dst_mem_view, const AffineMap& ind_src_map,
    const AffineMap& dir_src_map, const AffineMap& ind_dst_map,
    const AffineMap& dir_dst_map, const SmallVectorImpl<Value>& dir_src_indices,
    const SmallVectorImpl<Value>& dir_dst_indices, const IntegerSet time_set) {
  // Gather the operands for the new op. These are ordered as follows:
  //   <indices>, <time_symbols>, <multicast_info>
  SmallVector<Value, 16> operands;
  int num_ind_src_indices = 0, num_ind_dst_indices = 0;
  for (auto index : getIndirectSrcMapIndices()) {
    ++num_ind_src_indices;
    operands.push_back(index);
  }
  for (auto& index : dir_src_indices) operands.push_back(index);
  for (auto index : getIndirectDstMapIndices()) {
    ++num_ind_dst_indices;
    operands.push_back(index);
  }
  for (auto& index : dir_dst_indices) operands.push_back(index);

  for (auto sym : getTimeSymbols()) operands.push_back(sym);
  if (hasMulticastInfo()) operands.push_back(getMulticastInfo());

  // Create the new comp op in the innermost loop. Builder should already be set
  // to correct location.
  auto& op = *this;
  auto new_comp_op = CompositeIndirectLoadAndStoreOp::create(
      builder, op->getLoc(), ind_src_mem_view, dir_src_mem_view,
      ind_dst_mem_view, dir_dst_mem_view,
      getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
      ind_src_map, dir_src_map, ind_dst_map, dir_dst_map, operands,
      cast<VectorType>(getLoadInductionVarType()), getLoadSet().getValue(),
      getLoadOrder(), getStoreSet().getValue(), getStoreOrder(), time_set,
      getTimeOrder(),
      getLoadIndirectTimeAddrMap().has_value()
          ? getLoadIndirectTimeAddrMap().value()
          : builder.getEmptyAffineMap(),
      getLoadDirectTimeAddrMap(),
      getStoreIndirectTimeAddrMap().has_value()
          ? getStoreIndirectTimeAddrMap().value()
          : builder.getEmptyAffineMap(),
      getStoreDirectTimeAddrMap(), num_ind_src_indices, dir_src_indices.size(),
      num_ind_dst_indices, dir_dst_indices.size(), hasMulticastInfo() ? 1 : 0,
      getTimeSymbols().size());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = new_comp_op.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&new_comp_op.getRegion().front());
  IRMapping ir_map;
  for (auto& o : getRegion().getOps()) (void)builder.clone(*&o, ir_map);
  builder.restoreInsertionPoint(insert_pt);

  // Replace any uses of the old load_iv with the new one.
  auto orig_load_iv = getLoadInductionVar();
  auto new_load_iv = new_comp_op.getLoadInductionVar();
  for (auto& o : new_comp_op.getRegion().getOps())
    o.replaceUsesOfWith(orig_load_iv, new_load_iv);

  return new_comp_op;
}

//===----------------------------------------------------------------------===//
// CompositeIndirectLoadOp
//===----------------------------------------------------------------------===//

ParseResult CompositeIndirectLoadOp::parse(OpAsmParser& parser,
                                           OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType ind_memref_type, dir_memref_type;
  Type induction_var_type;
  OpAsmParser::UnresolvedOperand ind_memref_info, dir_memref_info,
      multicast_info;
  OpAsmParser::Argument induction_var;
  AffineMapAttr ind_map_attr, dir_map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> ind_map_operands,
      dir_map_operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> time_symbols_operands;
  // Parse region arguments.
  SmallVector<OpAsmParser::Argument, 1> region_args;
  SmallVector<Type, 1> arg_types;
  Region* body = result.addRegion();

  bool has_multicast = false;

  bool op_result = false;
  op_result = op_result || parser.parseKeyword("indirect") ||
              parser.parseColon() || parser.parseOperand(ind_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  ind_map_operands, ind_map_attr,
                  CompositeIndirectLoadOp::getIndirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  CompositeIndirectLoadOp::getDirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct";

  if (succeeded(parser.parseOptionalKeyword("multicast_info"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(multicast_info);
    has_multicast = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing multicast_info";

  op_result = op_result || parser.parseKeyword("time_symbols") ||
              parser.parseOperandList(time_symbols_operands,
                                      OpAsmParser::Delimiter::Paren) ||
              parser.parseComma() || parser.parseKeyword("load_iv") ||
              parser.parseLParen() || parser.parseArgument(induction_var) ||
              parser.parseColonType(induction_var_type) || parser.parseRParen();
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing time_symbols and ivs";

  // Induction variable.
  induction_var.type = induction_var_type;
  region_args.push_back(induction_var);
  arg_types.push_back(induction_var_type);

  result.attributes.push_back(builder.getNamedAttr(
      CompositeIndirectLoadOp::getOperandSegmentSizesAttrName(OperationName(
          CompositeIndirectLoadOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir_memref */, 1 /* dir_memref */,
           (int)ind_map_operands.size(), (int)dir_map_operands.size(),
           has_multicast ? 1 : 0, (int)time_symbols_operands.size()})));
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes) ||
              parser.parseRegion(*body, region_args);

  op_result =
      op_result || parser.parseColon() || parser.parseType(ind_memref_type) ||
      parser.parseComma() ||
      parser.resolveOperand(ind_memref_info, ind_memref_type, result.operands);
  op_result =
      op_result || parser.parseType(dir_memref_type) ||
      parser.resolveOperand(dir_memref_info, dir_memref_type, result.operands);

  if (op_result) return failure();

  // Adding map operands to the result operands after all the memref's.
  op_result = op_result || parser.resolveOperands(ind_map_operands, index_type,
                                                  result.operands);

  op_result = op_result || parser.resolveOperands(dir_map_operands, index_type,
                                                  result.operands);

  if (has_multicast)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  op_result = op_result || parser.resolveOperands(time_symbols_operands,
                                                  index_type, result.operands);

  std::optional<NamedAttribute> load_set_attr = result.attributes.getNamed(
      CompositeIndirectLoadOp::getLoadSetAttrName(result.name));
  if (!load_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "load set is missing in the operation";
  }
  auto load_set =
      mlir::dyn_cast<IntegerSetAttr>(load_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (load_set.getNumDims() < dir_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "load set and direct src array dimensions should match";
  }
  std::optional<NamedAttribute> load_order_attr = result.attributes.getNamed(
      CompositeIndirectLoadOp::getLoadOrderAttrName(result.name));
  if (load_order_attr.has_value()) {
    auto load_order =
        mlir::dyn_cast<AffineMapAttr>(load_order_attr->getValue()).getValue();
    if (load_set.getNumDims() != load_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "load set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "load order is missing in the operation";
  }

  return failure(op_result);
}

void CompositeIndirectLoadOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << " indirect:" << op.getIndirectMemref() << '[';
  if (AffineMapAttr ind_map_attr = getIndirectMapAttr(); ind_map_attr)
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr = getDirectMapAttr(); dir_map_attr)
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  if (hasMulticastInfo()) p << " multicast_info = " << op.getMulticastInfo();

  p << " time_symbols(";
  p.printOperands(op.getTimeSymbols());
  p << "), load_iv(" << op.getLoadInductionVar() << ':'
    << op.getLoadInductionVarType() << ')';
  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrName(), op.getDirectMapAttrName(),
                       op.getOperandSegmentSizesAttrName()});
  p.printNewline();
  p.printRegion(op.getRegion(), false, true);
  p << " : ";
  p << op.getIndirectMemrefType() << ", ";
  p << op.getDirectMemrefType();
}

void CompositeIndirectLoadOp::build(
    ::mlir::OpBuilder& builder, ::mlir::OperationState& result,
    Value indirect_memref, Value direct_memref, ::mlir::StringAttr dbgName,
    mlir::AffineMap indirect_map, mlir::AffineMap direct_map,
    ValueRange operands, Type type, IntegerSet load_set,
    mlir::AffineMap load_order, IntegerSet time_set, mlir::AffineMap time_order,
    mlir::AffineMap time_addr_map, uint32_t num_ind_memref_indices,
    uint32_t num_dir_memref_indices, uint32_t num_multicast_info,
    uint32_t num_time_symbols,
    CompositeIndirectLoadOp::BodyBuilderFn body_builder) {
  assert(((num_multicast_info == 0 || num_multicast_info == 1)) &&
         "expected either no multicast operand or a single SSA value");
  assert((num_ind_memref_indices > 0 && num_dir_memref_indices > 0) &&
         "indices are mandatory");

  assert((load_set.getNumDims() == load_order.getNumDims()) &&
         "The number of inputs in load set should match with number of "
         "inputs in load order");
  assert((load_order.getNumDims() == load_order.getNumResults()) &&
         "The number of inputs and results in load order should match "
         "with each other");
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == time_addr_map.getNumDims()) &&
         "The number of inputs in time set and load time address map should "
         "match with each other");

  result.addOperands(indirect_memref);
  result.addOperands(direct_memref);

  result.addOperands(operands);

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getIndirectMapAttrName(result.name),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrName(result.name),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getLoadSetAttrName(result.name),
                      IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderAttrName(result.name),
                      AffineMapAttr::get(load_order));
  result.addAttribute(getTimeSetAttrName(result.name),
                      IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderAttrName(result.name),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(time_addr_map));

  result.addAttribute(
      CompositeIndirectLoadOp::getOperandSegmentSizesAttrName(OperationName(
          CompositeIndirectLoadOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir_memref */, 1 /* dir_memref */,
           (int)num_ind_memref_indices, (int)num_dir_memref_indices,
           (int)num_multicast_info, (int)num_time_symbols}));

  // Add a body region with block arguments
  Region* body_region = result.addRegion();
  body_region->push_back(new Block);
  Block& body_block = body_region->front();
  // FIXME: which getLoc() should be used here?
  body_block.addArgument(type, direct_memref.getLoc());
  CompositeIndirectLoadOp::ensureTerminator(*body_region, builder,
                                            result.location);
}

//===----------------------------------------------------------------------===//
// CompositeIndirectStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeIndirectStoreOp::parse(OpAsmParser& parser,
                                            OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType ind_memref_type, dir_memref_type;
  OpAsmParser::UnresolvedOperand ind_memref_info, dir_memref_info,
      multicast_info;
  AffineMapAttr ind_map_attr, dir_map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> ind_map_operands,
      dir_map_operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> time_symbols_operands;
  // Parse region arguments.
  SmallVector<OpAsmParser::Argument, 1> region_args;
  SmallVector<Type, 1> arg_types;
  Region* body = result.addRegion();

  bool has_multicast = false;

  bool op_result = false;
  op_result = op_result || parser.parseKeyword("indirect") ||
              parser.parseColon() || parser.parseOperand(ind_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  ind_map_operands, ind_map_attr,
                  CompositeIndirectStoreOp::getIndirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  CompositeIndirectStoreOp::getDirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct";

  if (succeeded(parser.parseOptionalKeyword("multicast_info"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(multicast_info);
    has_multicast = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing multicast_info";

  op_result = op_result || parser.parseKeyword("time_symbols") ||
              parser.parseOperandList(time_symbols_operands,
                                      OpAsmParser::Delimiter::Paren);
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing time_symbols";

  result.attributes.push_back(builder.getNamedAttr(
      CompositeIndirectStoreOp::getOperandSegmentSizesAttrName(OperationName(
          CompositeIndirectStoreOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1, 1, (int)ind_map_operands.size(), (int)dir_map_operands.size(),
           has_multicast ? 1 : 0, (int)time_symbols_operands.size()})));
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes) ||
              parser.parseRegion(*body, region_args);

  op_result =
      op_result || parser.parseColon() || parser.parseType(ind_memref_type) ||
      parser.parseComma() ||
      parser.resolveOperand(ind_memref_info, ind_memref_type, result.operands);
  op_result =
      op_result || parser.parseType(dir_memref_type) ||
      parser.resolveOperand(dir_memref_info, dir_memref_type, result.operands);

  if (op_result) return failure();

  // Adding map operands to the result operands after all the memref's.
  op_result = op_result || parser.resolveOperands(ind_map_operands, index_type,
                                                  result.operands);

  op_result = op_result || parser.resolveOperands(dir_map_operands, index_type,
                                                  result.operands);

  if (has_multicast)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  op_result = op_result || parser.resolveOperands(time_symbols_operands,
                                                  index_type, result.operands);

  std::optional<NamedAttribute> store_set_attr = result.attributes.getNamed(
      CompositeIndirectStoreOp::getStoreSetAttrName(result.name));
  if (!store_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "store set is missing in the operation";
  }
  auto store_set =
      mlir::dyn_cast<IntegerSetAttr>(store_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (store_set.getNumDims() < dir_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "store set and direct array dimensions should match";
  }
  std::optional<NamedAttribute> store_order_attr = result.attributes.getNamed(
      CompositeIndirectStoreOp::getStoreOrderAttrName(result.name));
  if (store_order_attr.has_value()) {
    auto store_order =
        mlir::dyn_cast<AffineMapAttr>(store_order_attr->getValue()).getValue();
    if (store_set.getNumDims() != store_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "store set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "store order is missing in the operation";
  }

  return failure(op_result);
}

void CompositeIndirectStoreOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << " indirect:" << op.getIndirectMemref() << '[';
  if (AffineMapAttr ind_map_attr = getIndirectMapAttr(); ind_map_attr)
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr = getDirectMapAttr(); dir_map_attr)
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  p.printNewline();
  if (hasMulticastInfo()) p << " multicast_info = " << op.getMulticastInfo();

  p << " time_symbols(";
  p.printOperands(op.getTimeSymbols());
  p << ")";

  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrName(), op.getDirectMapAttrName(),
                       op.getOperandSegmentSizesAttrName()});
  p.printNewline();
  p.printRegion(op.getRegion(), false, true);
  p << " : ";
  p << op.getIndirectMemrefType() << ", ";
  p << op.getDirectMemrefType();
}

void CompositeIndirectStoreOp::build(
    ::mlir::OpBuilder& builder, ::mlir::OperationState& result,
    Value indirect_memref, Value direct_memref,
    /*optional*/ mlir::StringAttr dbgName, mlir::AffineMap indirect_map,
    mlir::AffineMap direct_map, ValueRange operands, Type type,
    IntegerSet store_set, mlir::AffineMap store_order, IntegerSet time_set,
    mlir::AffineMap time_order, mlir::AffineMap time_addr_map,
    uint32_t num_ind_memref_indices, uint32_t num_dir_memref_indices,
    uint32_t num_multicast_info, uint32_t num_time_symbols,
    CompositeIndirectStoreOp::BodyBuilderFn body_builder) {
  assert(((num_multicast_info == 0 || num_multicast_info == 1)) &&
         "expected either no multicast operand or a single SSA value");
  assert((num_ind_memref_indices > 0 && num_dir_memref_indices > 0) &&
         "indices are mandatory");

  assert((store_set.getNumDims() == store_order.getNumDims()) &&
         "The number of inputs in store set should match with number of "
         "inputs in store order");
  assert((store_order.getNumDims() == store_order.getNumResults()) &&
         "The number of inputs and results in store order should match "
         "with each other");
  assert((time_set.getNumDims() == time_order.getNumDims()) &&
         "The number of inputs in time set should match with number of "
         "inputs in time order");
  assert((time_order.getNumDims() == time_order.getNumResults()) &&
         "The number of inputs and results in time order should match "
         "with each other");
  assert((time_set.getNumDims() == time_addr_map.getNumDims()) &&
         "The number of inputs in time set and time address map should "
         "match with each other");

  result.addOperands(indirect_memref);
  result.addOperands(direct_memref);

  result.addOperands(operands);

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getIndirectMapAttrName(result.name),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrName(result.name),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getStoreOrderAttrName(result.name),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrName(result.name),
                      IntegerSetAttr::get(store_set));
  result.addAttribute(getTimeSetAttrName(result.name),
                      IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderAttrName(result.name),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrName(result.name),
                      AffineMapAttr::get(time_addr_map));

  result.addAttribute(
      CompositeIndirectLoadAndStoreOp::getOperandSegmentSizesAttrName(
          OperationName(CompositeIndirectStoreOp::getOperationName(),
                        builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir_memref */, 1 /* dir_memref */,
           (int)num_ind_memref_indices, (int)num_dir_memref_indices,
           (int)num_multicast_info, (int)num_time_symbols}));

  // Add a body region with block arguments
  Region* body_region = result.addRegion();
  body_region->push_back(new Block);
  Block& body_block = body_region->front();
  // FIXME: which getLoc() should be used here?
  body_block.addArgument(type, direct_memref.getLoc());
  CompositeIndirectStoreOp::ensureTerminator(*body_region, builder,
                                             result.location);
}

//===----------------------------------------------------------------------===//
// IndirectVectorLoadOp
//===----------------------------------------------------------------------===//

ParseResult IndirectVectorLoadOp::parse(OpAsmParser& parser,
                                        OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType ind_memref_type, dir_memref_type;
  Type result_type;
  OpAsmParser::UnresolvedOperand ind_memref_info, dir_memref_info,
      multicast_info;
  AffineMapAttr ind_map_attr, dir_map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> ind_map_operands,
      dir_map_operands;

  bool has_multicast = false;

  bool op_result = false;
  op_result = op_result || parser.parseKeyword("indirect") ||
              parser.parseColon() || parser.parseOperand(ind_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  ind_map_operands, ind_map_attr,
                  IndirectVectorLoadOp::getIndirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  IndirectVectorLoadOp::getDirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct";

  if (succeeded(parser.parseOptionalKeyword("multicast_info"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(multicast_info);
    has_multicast = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing multicast_info";

  result.attributes.push_back(builder.getNamedAttr(
      IndirectVectorLoadOp::getOperandSegmentSizesAttrName(OperationName(
          IndirectVectorLoadOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir */, 1 /* dir_memref */, (int)ind_map_operands.size(),
           (int)dir_map_operands.size(), has_multicast ? 1 : 0})));
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes);

  op_result =
      op_result || parser.parseColon() || parser.parseType(ind_memref_type) ||
      parser.parseComma() ||
      parser.resolveOperand(ind_memref_info, ind_memref_type, result.operands);
  op_result =
      op_result || parser.parseType(dir_memref_type) || parser.parseComma() ||
      parser.resolveOperand(dir_memref_info, dir_memref_type, result.operands);

  op_result = op_result || parser.parseType(result_type) ||
              parser.addTypeToList(result_type, result.types);

  if (op_result) return failure();

  // Adding map operands to the result operands after all the memref's.
  op_result = op_result || parser.resolveOperands(ind_map_operands, index_type,
                                                  result.operands);

  op_result = op_result || parser.resolveOperands(dir_map_operands, index_type,
                                                  result.operands);

  if (has_multicast)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  std::optional<NamedAttribute> load_set_attr = result.attributes.getNamed(
      IndirectVectorLoadOp::getLoadSetAttrName(result.name));
  if (!load_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "load set is missing in the operation";
  }
  auto load_set =
      mlir::dyn_cast<IntegerSetAttr>(load_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (load_set.getNumDims() < dir_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "load set and direct array dimensions should match";
  }
  std::optional<NamedAttribute> load_order_attr = result.attributes.getNamed(
      IndirectVectorLoadOp::getLoadOrderAttrName(result.name));
  if (load_order_attr.has_value()) {
    auto load_order =
        mlir::dyn_cast<AffineMapAttr>(load_order_attr->getValue()).getValue();
    if (load_set.getNumDims() != load_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "load set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "load order is missing in the operation";
  }

  return failure(op_result);
}

void IndirectVectorLoadOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << " indirect:" << op.getIndirectMemref() << '[';
  if (AffineMapAttr ind_map_attr = getIndirectMapAttr(); ind_map_attr)
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr = getDirectMapAttr(); dir_map_attr)
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  if (hasMulticastInfo()) p << " multicast_info = " << op.getMulticastInfo();

  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrName(), op.getDirectMapAttrName(),
                       op.getOperandSegmentSizesAttrName()});
  p.printNewline();
  p << " : ";
  p << op.getIndirectMemrefType() << ", ";
  p << op.getDirectMemrefType() << ", ";
  p << op.getType();
}

void IndirectVectorLoadOp::build(
    ::mlir::OpBuilder& builder, ::mlir::OperationState& result,
    Type result_type, Value indirect_memref, Value direct_memref,
    /*optional*/ mlir::StringAttr dbgName, mlir::AffineMap indirect_map,
    mlir::AffineMap direct_map, ValueRange operands, Type type,
    IntegerSet load_set, mlir::AffineMap load_order,
    uint32_t num_ind_memref_indices, uint32_t num_dir_memref_indices,
    uint32_t num_multicast_info) {
  assert(((num_multicast_info == 0 || num_multicast_info == 1)) &&
         "expected either no multicast operand or a single SSA value");
  assert((num_ind_memref_indices > 0 && num_dir_memref_indices > 0) &&
         "indices are mandatory");
  result.addOperands(indirect_memref);
  result.addOperands(direct_memref);

  result.addOperands(operands);

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getIndirectMapAttrName(result.name),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrName(result.name),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getLoadSetAttrName(result.name),
                      IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderAttrName(result.name),
                      AffineMapAttr::get(load_order));

  result.addAttribute(
      IndirectVectorLoadOp::getOperandSegmentSizesAttrName(OperationName(
          IndirectVectorLoadOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir_memref*/, 1 /* dir_memref */, (int)num_ind_memref_indices,
           (int)num_dir_memref_indices, (int)num_multicast_info}));
  result.types.push_back(result_type);
}

//===----------------------------------------------------------------------===//
// IndirectVectorStoreOp
//===----------------------------------------------------------------------===//
ParseResult IndirectVectorStoreOp::parse(OpAsmParser& parser,
                                         OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  Type value_to_store_type;
  MemRefType ind_memref_type, dir_memref_type;
  OpAsmParser::UnresolvedOperand value_to_store, ind_memref_info,
      dir_memref_info, multicast_info;
  AffineMapAttr ind_map_attr, dir_map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> ind_map_operands,
      dir_map_operands;

  bool has_multicast = false;

  bool op_result = false;
  op_result = op_result || parser.parseOperand(value_to_store);
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing value to store";

  op_result = op_result || parser.parseKeyword("indirect") ||
              parser.parseColon() || parser.parseOperand(ind_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  ind_map_operands, ind_map_attr,
                  IndirectVectorStoreOp::getIndirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  IndirectVectorStoreOp::getDirectMapAttrName(result.name),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct";

  if (succeeded(parser.parseOptionalKeyword("multicast_info"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(multicast_info);
    has_multicast = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing multicast_info";

  result.attributes.push_back(builder.getNamedAttr(
      IndirectVectorStoreOp::getOperandSegmentSizesAttrName(OperationName(
          IndirectVectorStoreOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr({1 /* value */, 1 /* indir*/, 1 /* dir */,
                                    (int)ind_map_operands.size(),
                                    (int)dir_map_operands.size(),
                                    has_multicast ? 1 : 0})));
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes);

  op_result = op_result || parser.parseColon() ||
              parser.parseType(value_to_store_type) || parser.parseComma() ||
              parser.resolveOperand(value_to_store, value_to_store_type,
                                    result.operands);
  op_result =
      op_result || parser.parseType(ind_memref_type) || parser.parseComma() ||
      parser.resolveOperand(ind_memref_info, ind_memref_type, result.operands);
  op_result =
      op_result || parser.parseType(dir_memref_type) ||
      parser.resolveOperand(dir_memref_info, dir_memref_type, result.operands);

  if (op_result) return failure();

  // Adding map operands to the result operands after all the memref's.
  op_result = op_result || parser.resolveOperands(ind_map_operands, index_type,
                                                  result.operands);

  op_result = op_result || parser.resolveOperands(dir_map_operands, index_type,
                                                  result.operands);

  if (has_multicast)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  std::optional<NamedAttribute> store_set_attr = result.attributes.getNamed(
      IndirectVectorStoreOp::getStoreSetAttrName(result.name));
  if (!store_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "store set is missing in the operation";
  }
  auto store_set =
      mlir::dyn_cast<IntegerSetAttr>(store_set_attr->getValue()).getValue();
  // TODO: enhance this with finding constant values and make sure they match.
  if (store_set.getNumDims() < dir_map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "store set and direct array dimensions should match";
  }
  std::optional<NamedAttribute> store_order_attr = result.attributes.getNamed(
      IndirectVectorStoreOp::getStoreOrderAttrName(result.name));
  if (store_order_attr.has_value()) {
    auto store_order =
        mlir::dyn_cast<AffineMapAttr>(store_order_attr->getValue()).getValue();
    if (store_set.getNumDims() != store_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "store set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "store order is missing in the operation";
  }

  return failure(op_result);
}

void IndirectVectorStoreOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << ' ' << op.getValue();
  p << " indirect:" << op.getIndirectMemref() << '[';
  if (AffineMapAttr ind_map_attr = getIndirectMapAttr(); ind_map_attr)
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr = getDirectMapAttr(); dir_map_attr)
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  p.printNewline();
  if (hasMulticastInfo()) {
    p << " multicast_info = " << op.getMulticastInfo();
    p.printNewline();
  }
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrName(), op.getDirectMapAttrName(),
                       op.getOperandSegmentSizesAttrName()});
  p.printNewline();
  p << " : ";
  if (auto vtype = op.getVectorType())
    p << vtype;
  else if (auto custom_vtype =
               dyn_cast<dataflow::CustomVectorType>(getValue().getType()))
    p << custom_vtype;
  p << ", " << op.getIndirectMemrefType() << ", ";
  p << op.getDirectMemrefType();
}

void IndirectVectorStoreOp::build(
    ::mlir::OpBuilder& builder, ::mlir::OperationState& result, Value value,
    Value indirect_memref, Value direct_memref,
    /*optional*/ mlir::StringAttr dbgName, mlir::AffineMap indirect_map,
    mlir::AffineMap direct_map, ValueRange operands, Type type,
    IntegerSet store_set, mlir::AffineMap store_order,
    uint32_t num_ind_memref_indices, uint32_t num_dir_memref_indices,
    uint32_t num_multicast_info) {
  assert(((num_multicast_info == 0 || num_multicast_info == 1)) &&
         "expected either no multicast operand or a single SSA value");
  assert((num_ind_memref_indices > 0 && num_dir_memref_indices > 0) &&
         "indices are mandatory");
  result.addOperands(value);
  result.addOperands(indirect_memref);
  result.addOperands(direct_memref);

  result.addOperands(operands);

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getIndirectMapAttrName(result.name),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrName(result.name),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getStoreOrderAttrName(result.name),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrName(result.name),
                      IntegerSetAttr::get(store_set));

  result.addAttribute(
      IndirectVectorStoreOp::getOperandSegmentSizesAttrName(OperationName(
          IndirectVectorStoreOp::getOperationName(), builder.getContext())),
      builder.getDenseI32ArrayAttr(
          {1 /* indir_memref*/, 1 /* dir_memref */, (int)num_ind_memref_indices,
           (int)num_dir_memref_indices, (int)num_multicast_info}));
}

//===----------------------------------------------------------------------===//
// SymbolicVectorLoadOp
//===----------------------------------------------------------------------===//
// Syntax:
//   %0 = agen.symbolic_vector_load
//            %memref[indices:(%1, %2), strides:(%3, %4)], multicast_info(%5) :
//            memref<128xi8>, vector<128xi8>
ParseResult SymbolicVectorLoadOp::parse(OpAsmParser& parser,
                                        OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType memref_type;
  Type result_type;
  OpAsmParser::UnresolvedOperand memref_info, multicast_info;
  SmallVector<OpAsmParser::UnresolvedOperand> indices, strides;
  auto op_result =
      parser.parseOperand(memref_info) || parser.parseLSquare() ||
      parser.parseKeyword("indices") || parser.parseColon() ||
      parser.parseOperandList(indices, OpAsmParser::Delimiter::Paren);

  op_result = op_result || parser.parseComma() ||
              parser.parseKeyword("strides") || parser.parseColon() ||
              parser.parseOperandList(strides, OpAsmParser::Delimiter::Paren);

  op_result = op_result || parser.parseRSquare();

  bool has_multicast_info = false;
  if (!op_result && succeeded(parser.parseOptionalComma())) {
    op_result = op_result || parser.parseKeyword("multicast_info") ||
                parser.parseLParen() || parser.parseOperand(multicast_info) ||
                parser.parseRParen();
    has_multicast_info = true;
  }

  op_result = op_result || parser.parseOptionalAttrDict(result.attributes) ||
              parser.parseColonType(memref_type) || parser.parseComma() ||
              parser.parseType(result_type) ||
              parser.resolveOperand(memref_info, memref_type, result.operands);

  op_result = op_result ||
              parser.resolveOperands(indices, index_type, result.operands) ||
              parser.resolveOperands(strides, index_type, result.operands) ||
              parser.addTypeToList(result_type, result.types);

  if (has_multicast_info)
    op_result = op_result || parser.resolveOperand(multicast_info, index_type,
                                                   result.operands);

  IntegerAttr num_indices_attr = builder.getI32IntegerAttr(indices.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorLoadOp::getNumIndicesAttrName(result.name),
      num_indices_attr));

  IntegerAttr num_strides_attr = builder.getI32IntegerAttr(strides.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorLoadOp::getNumStridesAttrName(result.name),
      num_strides_attr));

  return failure(op_result);
}

void SymbolicVectorLoadOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << " " << op.getMemref() << '[';
  p << "indices:(";

  auto indices = op.getIndices();
  assert(indices.size() > 0);
  p << indices.front();

  for (unsigned i = 1, e = indices.size(); i < e; ++i) p << ", " << indices[i];
  p << "), strides:(";

  auto strides = op.getStrides();
  assert(strides.size() > 0);
  p << strides.front();

  for (unsigned i = 1, e = strides.size(); i < e; ++i) p << ", " << strides[i];
  p << ")]";

  if (op.getMulticastInfo().has_value())
    p << ", multicast_info(" << op.getMulticastInfo().value() << ")";

  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs*/ {op.getNumIndicesAttrName(),
                       SymbolicVectorLoadOp::getNumStridesAttrName()});
  p << " : " << op.getMemref().getType() << ", " << op.getType();
}

LogicalResult SymbolicVectorLoadOp::verify() {
  auto& op = *this;

  // Memref should be one dimensional.
  auto mem_ref_type = dyn_cast<MemRefType>(op.getMemref().getType());
  if (!mem_ref_type) return emitOpError("memref is not of MemRefType");
  if (mem_ref_type.getShape().size() != 1)
    return emitOpError("memref is not one dimensional");

  auto mem_ref_op =
      op.getMemref().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  if (!mem_ref_op)
    return emitOpError("memref should be a dataflow.get_logical_memory_view");
  if (!mem_ref_op.getLayoutMap().isIdentity())
    return emitOpError("memref layout map should be an identity map");

  // Result type should be one dimensional vector.
  auto result_vtype = op.getVectorType();
  auto result_custom_vtype =
      dyn_cast<dataflow::CustomVectorType>(getResult().getType());
  assert(result_vtype || result_custom_vtype);
  if (!result_custom_vtype && result_vtype.getShape().size() != 1)
    return emitOpError("result type is not one dimensional");

  // Memref type and result type should have the same element type.
  auto elem_type = result_vtype ? result_vtype.getElementType()
                                : result_custom_vtype.getElementType();
  if (elem_type != mem_ref_type.getElementType())
    return emitOpError("result type does not match memref element type");

  // There should be one or more indices.
  auto indices_size = op.getIndices().size();
  if (indices_size < 1) return emitOpError("indices should not be empty");

  // There should be an equal amount of strides to the indices.
  if (op.getStrides().size() != indices_size)
    return emitOpError("strides should be the same size as indices");

  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicVectorStoreOp
//===----------------------------------------------------------------------===//
// Syntax:
//   agen.symbolic_vector_store %data,
//       %memref[indices:(%0, %1), strides:(%2, %3)] :
//       memref<128xi8>, vector<128xi8>
ParseResult SymbolicVectorStoreOp::parse(OpAsmParser& parser,
                                         OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType memref_type;
  Type result_type;
  OpAsmParser::UnresolvedOperand storeValueInfo;
  OpAsmParser::UnresolvedOperand memref_info;
  SmallVector<OpAsmParser::UnresolvedOperand> indices, strides;
  auto op_result =
      parser.parseOperand(storeValueInfo) || parser.parseComma() ||
      parser.parseOperand(memref_info) || parser.parseLSquare() ||
      parser.parseKeyword("indices") || parser.parseColon() ||
      parser.parseOperandList(indices, OpAsmParser::Delimiter::Paren);

  op_result = op_result || parser.parseComma() ||
              parser.parseKeyword("strides") || parser.parseColon() ||
              parser.parseOperandList(strides, OpAsmParser::Delimiter::Paren);

  op_result = op_result || parser.parseRSquare();

  op_result =
      op_result || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memref_type) || parser.parseComma() ||
      parser.parseType(result_type) ||
      parser.resolveOperand(storeValueInfo, result_type, result.operands) ||
      parser.resolveOperand(memref_info, memref_type, result.operands);

  op_result = op_result ||
              parser.resolveOperands(indices, index_type, result.operands) ||
              parser.resolveOperands(strides, index_type, result.operands);

  IntegerAttr num_indices_attr = builder.getI32IntegerAttr(indices.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorStoreOp::getNumIndicesAttrName(result.name),
      num_indices_attr));

  IntegerAttr num_strides_attr = builder.getI32IntegerAttr(strides.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorStoreOp::getNumStridesAttrName(result.name),
      num_strides_attr));

  return failure(op_result);
}

void SymbolicVectorStoreOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << ' ' << op.getValueToStore();
  p << ", " << op.getMemref() << '[';
  p << "indices:(";

  auto indices = op.getIndices();
  assert(indices.size() > 0);
  p << indices.front();

  for (unsigned i = 1, e = indices.size(); i < e; ++i) {
    p << ", " << indices[i];
  }
  p << "), strides:(";

  auto strides = op.getStrides();
  assert(strides.size() > 0);
  p << strides.front();

  for (unsigned i = 1, e = strides.size(); i < e; ++i) {
    p << ", " << strides[i];
  }
  p << ")]";

  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs*/ {op.getNumIndicesAttrName(),
                       SymbolicVectorStoreOp::getNumStridesAttrName()});
  p << " : " << op.getMemref().getType() << ", "
    << op.getValueToStore().getType();
}

LogicalResult SymbolicVectorStoreOp::verify() {
  auto& op = *this;

  // Memref should be one dimensional.
  auto mem_ref_type = dyn_cast<MemRefType>(op.getMemref().getType());
  if (!mem_ref_type) return emitOpError("memref is not of MemRefType");
  if (mem_ref_type.getShape().size() != 1)
    return emitOpError("memref is not one dimensional");

  auto mem_ref_op =
      op.getMemref().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  if (!mem_ref_op)
    return emitOpError("memref should be a dataflow.get_logical_memory_view");
  if (!mem_ref_op.getLayoutMap().isIdentity())
    return emitOpError("memref layout map should be an identity map");

  // Result type should be one dimensional vector.
  auto result_vtype = op.getVectorType();
  auto result_custom_vtype =
      dyn_cast<dataflow::CustomVectorType>(getValueToStore().getType());

  if (!result_custom_vtype && result_vtype.getShape().size() != 1)
    return emitOpError("result type is not one dimensional");

  // Memref type and result type should have the same element type.
  auto elem_type = result_vtype ? result_vtype.getElementType()
                                : result_custom_vtype.getElementType();
  if (elem_type != mem_ref_type.getElementType())
    return emitOpError("result type does not match memref element type");

  // There should be one or more indices.
  if (op.getIndices().empty())
    return emitOpError("indices should not be empty");

  // There should be an equal amount of strides to the indices.
  if (op.getStrides().size() != op.getIndices().size())
    return emitOpError("strides should be the same size as indices");

  return success();
}

//===----------------------------------------------------------------------===//
// CompositeMemoryInterleaveOp
//===----------------------------------------------------------------------===//
ParseResult CompositeMemoryInterleaveOp::parse(OpAsmParser& parser,
                                               OperationState& result) {
  auto& builder = parser.getBuilder();
  Region* interleaveRegion = result.addRegion();

  // Parse the attributes
  auto op_result = parser.parseOptionalAttrDict(result.attributes) ||
                   parser.parseRegion(*interleaveRegion, {});

  CompositeMemoryInterleaveOp::ensureTerminator(*interleaveRegion, builder,
                                                result.location);

  return failure(op_result);
}

void CompositeMemoryInterleaveOp::print(OpAsmPrinter& p) {
  auto& op = *this;

  if (op->getAttrs().empty())
    p << " {}";
  else
    p.printOptionalAttrDict(op->getAttrs());

  p.printRegion(op.getRegion(),
                /*printEntryBlockArgs*/ false,
                /*printBlockTerminators*/ false);
}

void CompositeMemoryInterleaveOp::build(::mlir::OpBuilder& builder,
                                        ::mlir::OperationState& result,
                                        /*optional*/ mlir::StringAttr dbgName,
                                        uint32_t granularity) {
  assert((granularity > 0) && "expected granularity greater than 0");

  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  // Add granularity attribute
  result.addAttribute(getGranularityAttrName(result.name),
                      builder.getI32IntegerAttr(granularity));

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeMemoryInterleaveOp::ensureTerminator(*bodyRegion, builder,
                                                result.location);
}

void CompositeMemoryInterleaveOp::build(::mlir::OpBuilder& builder,
                                        ::mlir::OperationState& result,
                                        /*optional*/ mlir::StringAttr dbgName) {
  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeMemoryInterleaveOp::ensureTerminator(*bodyRegion, builder,
                                                result.location);
}

//===----------------------------------------------------------------------===//
// SetTransferMaskStateOp
//===----------------------------------------------------------------------===//
ParseResult SetTransferMaskStateOp::parse(OpAsmParser& parser,
                                          OperationState& result) {
  auto& builder = parser.getBuilder();
  auto loc = parser.getCurrentLocation();

  // Parse the operands.
  OpAsmParser::UnresolvedOperand mask_value;
  auto op_result = parser.parseKeyword("mask_value") || parser.parseLParen() ||
                   parser.parseOperand(mask_value) || parser.parseRParen();

  // Parse the attributes. Default attribute parsing will handle the attrs
  // presented in the default manner. Attributes such as the individual masks
  // need custom parsing.
  op_result = op_result || parser.parseOptionalAttrDict(result.attributes);

  // Collect the mask suffixes and their corresponding values. These will be
  // properly ordered for the operation attributes afterwards.
  std::map<char, int> unmasked_elems_map, masked_elems_map;
  char start_mask_id = 'A';
  const auto prefix_len = getMaskPrefixAttrStrName().size();
  for (auto& attr : result.attributes) {
    // Find the "mask*"" attributes and add their unmasked/masked values to
    // unmasked_elems_map/masked_elems_map.
    auto attr_keyword = attr.getName().strref();
    if (attr_keyword.starts_with(getMaskPrefixAttrStrName())) {
      if (attr_keyword.size() <= prefix_len)
        return parser.emitError(loc, "mask attribute is not properly labeled");

      // Get the mask id. For example, maskA has a mask_id of 'A'.
      char mask_id = attr_keyword[prefix_len];
      auto attr_value = attr.getValue();
      auto attr_stringattr = dyn_cast<StringAttr>(attr_value);
      if (!attr_stringattr)
        return parser.emitError(
            loc, "mask attribute does not describe the mask pattern");

      // Collect unmasked and masked elements from the attr_string. The string
      // should be in format similar to:
      //   maskA = (unmasked = 1 : i32, masked = 1 : i32)
      // Remove the spaces for easier parsing.
      auto attr_string = attr_stringattr.getValue().str();
      auto attr_string_nospace =
          std::remove(attr_string.begin(), attr_string.end(), ' ');
      attr_string.erase(attr_string_nospace, attr_string.end());

      // Collect unmasked info.
      auto unmasked_start = attr_string.find('=');
      auto unmasked_end = attr_string.find(',');
      if (unmasked_start == std::string::npos ||
          unmasked_end == std::string::npos)
        return parser.emitError(
            loc, "cannot determine where unmasked/masked values are");
      unmasked_elems_map[mask_id] = std::stoi(attr_string.substr(
          unmasked_start + 1, unmasked_end - unmasked_start));

      // Collect masked info.
      auto masked_start = attr_string.find('=', unmasked_end);
      auto masked_end = attr_string.find(')');
      if (unmasked_start == std::string::npos ||
          unmasked_end == std::string::npos)
        return parser.emitError(
            loc, "cannot determine where unmasked/masked values are");
      masked_elems_map[mask_id] = std::stoi(
          attr_string.substr(masked_start + 1, masked_end - unmasked_start));
    }
  }
  // Order the mask information into the appropriate attributes. This will be
  // alphabetical based on mask_id (map key).
  auto orderMaskInfo = [](std::map<char, int>& info_map,
                          std::vector<int>& num_elems, char start_mask_id) {
    for (auto& entry : info_map) {
      int idx = entry.first - start_mask_id;
      assert((static_cast<std::size_t>(idx) < info_map.size()) &&
             "encountered an unexpected mask_id");
      num_elems[idx] = entry.second;
    }
  };
  std::vector<int> num_unmasked_elems(unmasked_elems_map.size()),
      num_masked_elems(masked_elems_map.size());
  orderMaskInfo(unmasked_elems_map, num_unmasked_elems, start_mask_id);
  orderMaskInfo(masked_elems_map, num_masked_elems, start_mask_id);

  // If masks were detected, add the attributes. Only have to check if
  // num_unmasked_elems is empty or not because when num_unmasked_elems is
  // filled, num_masked_elems is required and is filled.
  if (!num_unmasked_elems.empty()) {
    result.addAttribute("num_unmasked_elements",
                        builder.getI32ArrayAttr(num_unmasked_elems));
    result.addAttribute("num_masked_elements",
                        builder.getI32ArrayAttr(num_masked_elems));
  }

  // Parse the operands
  auto index_type = builder.getIndexType();
  op_result = op_result || parser.parseColonType(index_type) ||
              parser.resolveOperand(mask_value, index_type, result.operands);

  // Parse the result type
  Type result_type;
  op_result = op_result || parser.parseComma() ||
              parser.parseType(result_type) ||
              parser.addTypeToList(result_type, result.types);

  return failure(op_result);
}

void SetTransferMaskStateOp::print(OpAsmPrinter& p) {
  auto& op = *this;

  // Print the operands
  p << " mask_value(" << op.getMaskValue() << ") ";

  // Print the attributes
  p << "{ num_slices = " << op.getNumSlicesAttr();
  p << ", slice_mask_map = \"" << op.getSliceMaskMap() << "\"";

  // If the operation has masks defined, print them in a readable format.
  // Example:
  //   maskA = (unmasked = 1 : i32, masked = 1 : i32)
  if (op.getNumUnmaskedElements().has_value() &&
      op.getNumMaskedElements().has_value()) {
    auto num_unmasked_elems = op.getNumUnmaskedElements().value();
    auto num_masked_elems = op.getNumMaskedElements().value();
    assert((num_unmasked_elems.size() == num_masked_elems.size()) &&
           "expecting num unmasked/masked elements to be the same");
    char mask_id = 'A';
    for (auto [unmasked, masked] :
         llvm::zip(num_unmasked_elems, num_masked_elems)) {
      p << ", mask" << mask_id << " = \"(unmasked = " << unmasked
        << ", masked = " << masked << ")\"";
      ++mask_id;
    }
  }

  p << " } :  " << getMaskValueType() << " , " << op.getType();
}

LogicalResult SetTransferMaskStateOp::verify() {
  auto& op = *this;
  // Check the number of masked/unmasked values match
  if (op.getNumUnmaskedElements().has_value()) {
    if (!op.getNumMaskedElements().has_value())
      return op->emitOpError(
          "cannot have unmasked values if there are no masked values");
    if (op.getNumUnmaskedElements().value().size() !=
        op.getNumMaskedElements().value().size())
      return op->emitOpError(
          "number of unmasked values and number of masked values must match");
  } else if (op.getNumMaskedElements().has_value())
    return op->emitOpError(
        "cannot have masked values if there are no unmasked values");

  return success();
}

// Returns true if the slice_mask_map reflects a reset/unmask pattern. The
// reset/unmask pattern is:
// (0)(0)(0)(0)(0)(0)(0)(0)
bool SetTransferMaskStateOp::isUnmask() {
  std::regex pattern("^(\\(0\\)){8}$");
  return std::regex_match(getSliceMaskMap().str(), pattern);
}

// Returns true if the slice_mask_map reflects a full mask pattern. The
// full mask pattern is:
// (1)(1)(1)(1)(1)(1)(1)(1)
bool SetTransferMaskStateOp::isFullMask() {
  std::regex pattern("^(\\(1\\)){8}$");
  return std::regex_match(getSliceMaskMap().str(), pattern);
}
