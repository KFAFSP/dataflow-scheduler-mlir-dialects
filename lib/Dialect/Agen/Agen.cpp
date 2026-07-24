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
  return VectorLoadOp::create(builder, getLoc(), getResult().getType(),
                              mem_view, getDbgNameAttr(), subscripts_map,
                              indices, getLoadSet(), getLoadOrder(),
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
  return VectorStoreOp::create(builder, getLoc(), getValueToStore(), mem_view,
                               getDbgNameAttr(), getAffineMap(), indices,
                               getStoreSet(), getStoreOrder());
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
  OpAsmParser::UnresolvedOperand src_mem_ref;
  AffineMapAttr src_map;
  SmallVector<OpAsmParser::UnresolvedOperand> src_operands;
  if (parser.parseKeyword("src") || parser.parseColon() ||
      parser.parseOperand(src_mem_ref) ||
      parseAffineMapOfSSAIds(parser, src_map, src_operands)) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand dst_mem_ref;
  AffineMapAttr dst_map;
  SmallVector<OpAsmParser::UnresolvedOperand> dst_operands;
  if (parser.parseKeyword("dst") || parser.parseColon() ||
      parser.parseOperand(dst_mem_ref) ||
      parseAffineMapOfSSAIds(parser, dst_map, dst_operands)) {
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

  result.addAttribute(getSrcMapAttrName(result.name), src_map);
  result.addAttribute(getDstMapAttrName(result.name), dst_map);
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, static_cast<int32_t>(src_operands.size()),
                           static_cast<int32_t>(dst_operands.size()),
                           static_cast<int32_t>(time_symbols.size()),
                           multicast_info.location.isValid() ? 1 : 0}));
  return success();
}

void CompositeLoadAndStoreOp::print(OpAsmPrinter& p) {
  p << " src:" << getSrcMemRef();
  printAffineMapOfSSAIds(p, *this, getSrcMapAttr(), getSrcOperands());
  p << " dst:" << getDstMemRef();
  printAffineMapOfSSAIds(p, *this, getDstMapAttr(), getDstOperands());

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
    ValueRange src_operands, AffineMapAttr dst_map, ValueRange dst_operands,
    IntegerSetAttr load_set, AffineMapAttr load_order, IntegerSetAttr store_set,
    AffineMapAttr store_order, ValueRange time_symbols, IntegerSetAttr time_set,
    AffineMapAttr time_order, AffineMapAttr load_time_addr_map,
    AffineMapAttr store_time_addr_map, VectorType type,
    AgenRoutingDirectionAttr dir, Value multicast_info,
    function_ref<void(OpBuilder&, Location, Value)> body_builder) {
  state.addOperands({src_mem_ref, dst_mem_ref});
  state.addOperands(src_operands);
  state.addOperands(dst_operands);
  state.addOperands(time_symbols);
  if (multicast_info) {
    state.operands.push_back(multicast_info);
  }

  state.addAttribute(getSrcMapAttrName(state.name), src_map);
  state.addAttribute(getDstMapAttrName(state.name), dst_map);
  state.addAttribute(getLoadSetAttrName(state.name), load_set);
  state.addAttribute(getLoadOrderAttrName(state.name), load_order);
  state.addAttribute(getStoreSetAttrName(state.name), store_set);
  state.addAttribute(getStoreOrderAttrName(state.name), store_order);
  state.addAttribute(getTimeSetAttrName(state.name), time_set);
  state.addAttribute(getTimeOrderAttrName(state.name), time_order);
  state.addAttribute(getLoadTimeAddrMapAttrName(state.name),
                     load_time_addr_map);
  state.addAttribute(getStoreTimeAddrMapAttrName(state.name),
                     store_time_addr_map);
  if (dbg_name) {
    state.addAttribute(getDbgNameAttrName(state.name), dbg_name);
  }
  if (dir) {
    state.addAttribute(getDirAttrName(state.name), dir);
  }

  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(src_operands.size()),
           static_cast<int32_t>(dst_operands.size()),
           static_cast<int32_t>(time_symbols.size()), multicast_info ? 1 : 0}));

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
      builder, getLoc(), src_mem_view, dst_mem_view, getDbgNameAttr(),
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
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType memref_type;
  Type inductionVar_type;
  OpAsmParser::UnresolvedOperand memref_info;
  OpAsmParser::Argument inductionVariable;
  AffineMapAttr map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> map_operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> time_symbols_operands;
  // Parse region arguments.
  SmallVector<OpAsmParser::Argument, 1> regionArgs;
  SmallVector<Type, 1> argTypes;
  Region* body = result.addRegion();

  auto op_result =
      parser.parseOperand(memref_info) ||
      parser.parseAffineMapOfSSAIds(map_operands, map_attr,
                                    CompositeLoadOp::getMapAttrStrName(),
                                    result.attributes) ||
      parser.parseKeyword("time_symbols") ||
      parser.parseOperandList(time_symbols_operands,
                              OpAsmParser::Delimiter::Paren) ||
      parser.parseLParen() || parser.parseArgument(inductionVariable) ||
      parser.parseColonType(inductionVar_type) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes);

  // Induction variable.
  inductionVariable.type = inductionVar_type;
  regionArgs.push_back(inductionVariable);
  argTypes.push_back(inductionVar_type);

  op_result =
      op_result || parser.parseRegion(*body, regionArgs) ||
      parser.parseColonType(memref_type) ||
      parser.resolveOperand(memref_info, memref_type, result.operands) ||
      parser.resolveOperands(map_operands, index_type, result.operands) ||
      parser.resolveOperands(time_symbols_operands, index_type,
                             result.operands);

  IntegerAttr num_memref_indices_attr =
      builder.getI32IntegerAttr(map_operands.size());
  result.attributes.push_back(
      builder.getNamedAttr(CompositeLoadOp::getNumMemrefIndicesAttrStrName(),
                           num_memref_indices_attr));

  std::optional<NamedAttribute> load_set_attr =
      result.attributes.getNamed(CompositeLoadOp::getLoadSetAttrStrName());
  if (!load_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "Load set is missing in the operation";
  }

  auto load_set =
      mlir::dyn_cast<IntegerSetAttr>(load_set_attr->getValue()).getValue();

  // TODO: enhance this with finding constant values and make sure they match.
  if (load_set.getNumDims() < map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "Load set and Array dimensions should match";
  }

  std::optional<NamedAttribute> load_order_attr =
      result.attributes.getNamed(CompositeLoadOp::getLoadOrderMapAttrStrName());
  if (load_order_attr.has_value()) {
    auto load_order =
        mlir::dyn_cast<AffineMapAttr>(load_order_attr->getValue()).getValue();
    if (load_set.getNumDims() != load_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "Load set and order dimensions should match";
    }
  } else {
    return parser.emitError(parser.getNameLoc())
           << "Load order is missing in the operation";
  }

  return failure(op_result);
}

void CompositeLoadOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << ' ' << op.getMemRef() << '[';
  if (AffineMapAttr map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getMapAttrStrName()))
    p.printAffineMapOfSSAIds(map_attr, op.getMapIndices());
  p << ']';
  p.printNewline();
  p << " time_symbols(";
  p.printOperands(op.getTimeSymbols());
  p << ')';
  p << '(' << op.getLoadInductionVar() << ':' << op.getLoadInductionVarType()
    << ')';
  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getMapAttrStrName(),
                       op.getNumMemrefIndicesAttrStrName()});
  p.printNewline();
  p.printRegion(op.getRegion(), false, true);
  p << " : " << op.getMemRefType();
}

void CompositeLoadOp::build(OpBuilder& builder, OperationState& result,
                            Value memref, mlir::StringAttr dbgName,
                            AffineMap map, ValueRange operands, Type load_type,
                            IntegerSet load_set, AffineMap load_order,
                            IntegerSet time_set, AffineMap time_order,
                            AffineMap time_addr_map,
                            uint32_t num_memref_indices,
                            CompositeLoadOp::BodyBuilderFn bodyBuilder) {
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
  result.addOperands(operands);
  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getLoadSetAttrStrName(), IntegerSetAttr::get(load_set));
  result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  result.addAttribute(getLoadOrderMapAttrStrName(),
                      AffineMapAttr::get(load_order));
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(time_addr_map));
  result.addAttribute(getNumMemrefIndicesAttrStrName(),
                      builder.getI32IntegerAttr(num_memref_indices));
  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block& bodyBlock = bodyRegion->front();
  // FIXME: which getLoc() should be used here?
  bodyBlock.addArgument(load_type, memref.getLoc());
  CompositeLoadOp::ensureTerminator(*bodyRegion, builder, result.location);
}

CompositeLoadOp CompositeLoadOp::cloneWithNewAccessInfo(
    OpBuilder& builder, const Value& mem_view, const AffineMap& subscripts_map,
    const SmallVectorImpl<Value>& indices, const IntegerSet& time_set) {
  // Gather the operands for the new op. These are ordered as follows:
  //   <indices>, <time symbols>
  SmallVector<Value, 16> operands;
  for (auto& index : indices) operands.push_back(index);
  int remaining_operands_idx = getNumMemrefIndices();
  for (std::size_t i = remaining_operands_idx; i < getOperands1().size(); ++i)
    operands.emplace_back(getOperands1()[i]);

  // Create the new comp op in the innermost loop. Builder should already be set
  // to correct location.
  auto& op = *this;
  auto new_comp_load = CompositeLoadOp::create(
      builder, op->getLoc(), mem_view,
      getDbgName().has_value() ? builder.getStringAttr(getDbgName().value())
                               : builder.getStringAttr(""),
      subscripts_map, operands, cast<VectorType>(getLoadInductionVarType()),
      getLoadSet().getValue(), getLoadOrder(), time_set, getTimeOrder(),
      getTimeAddrMap(), indices.size());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = new_comp_load.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&new_comp_load.getRegion().front());
  IRMapping ir_map;
  for (auto& o : getRegion().getOps()) (void)builder.clone(*&o, ir_map);
  builder.restoreInsertionPoint(insert_pt);

  // Replace any uses of the old load_iv with the new one.
  auto orig_load_iv = getLoadInductionVar();
  auto new_load_iv = new_comp_load.getLoadInductionVar();
  for (auto& o : new_comp_load.getRegion().getOps())
    o.replaceUsesOfWith(orig_load_iv, new_load_iv);

  return new_comp_load;
}

//===----------------------------------------------------------------------===//
// CompositeStoreOp
//===----------------------------------------------------------------------===//

ParseResult CompositeStoreOp::parse(OpAsmParser& parser,
                                    OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  MemRefType memref_type;
  Type input_vector_type;
  OpAsmParser::UnresolvedOperand memref_info, input_vector;
  bool getHaveInputVector = false;
  AffineMapAttr map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> map_operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> time_symbols_operands;
  // Parse region arguments.
  SmallVector<OpAsmParser::Argument, 1> regionArgs;
  SmallVector<Type, 1> argTypes;
  Region* body = result.addRegion();

  // parse memref
  auto op_result = parser.parseOperand(memref_info) ||
                   parser.parseAffineMapOfSSAIds(
                       map_operands, map_attr,
                       CompositeLoadOp::getMapAttrStrName(), result.attributes);
  // parse optional input vector for coalesce store
  if (succeeded(parser.parseOptionalKeyword("input_vector"))) {
    op_result =
        op_result || parser.parseEqual() || parser.parseOperand(input_vector);
    result.addAttribute(CompositeStoreOp::getHaveInputVectorAttrStrName(),
                        builder.getBoolAttr(true));
    getHaveInputVector = true;
  }
  // parse time symbols and attributes
  op_result = op_result || parser.parseKeyword("time_symbols") ||
              parser.parseOperandList(time_symbols_operands,
                                      OpAsmParser::Delimiter::Paren) ||
              parser.parseOptionalAttrDict(result.attributes);
  // parse optional region
  auto region_parse_result = parser.parseOptionalRegion(*body, regionArgs);
  if (region_parse_result.has_value()) {
    // check if attr exists
    if (getHaveInputVector) {
      return parser.emitError(parser.getNameLoc())
             << "for composite_store, input_vector and region can't co-exist";
    }
    op_result = op_result || region_parse_result.value();
    result.addAttribute(CompositeStoreOp::getHaveInputVectorAttrStrName(),
                        builder.getBoolAttr(false));
  }
  CompositeStoreOp::ensureTerminator(*body, builder, result.location);
  // parse memref type
  op_result = op_result || parser.parseColonType(memref_type);
  // parse optional input_vector type
  if (getHaveInputVector) {
    op_result =
        op_result || parser.parseComma() || parser.parseType(input_vector_type);
  }

  // resolve operands
  op_result =
      op_result ||
      parser.resolveOperand(memref_info, memref_type, result.operands) ||
      parser.resolveOperands(map_operands, index_type, result.operands);
  if (getHaveInputVector) {
    op_result =
        op_result ||
        parser.resolveOperand(input_vector, input_vector_type, result.operands);
  }
  op_result = op_result || parser.resolveOperands(time_symbols_operands,
                                                  index_type, result.operands);

  IntegerAttr num_memref_indices_attr =
      builder.getI32IntegerAttr(map_operands.size());
  result.attributes.push_back(
      builder.getNamedAttr(CompositeLoadOp::getNumMemrefIndicesAttrStrName(),
                           num_memref_indices_attr));

  // make sure getHaveInputVector attribute exists
  std::optional<NamedAttribute> getHaveInputVector_attr =
      result.attributes.getNamed(
          CompositeStoreOp::getHaveInputVectorAttrStrName());
  if (!getHaveInputVector_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "composite store requires getHaveInputVector attribute";
  }

  std::optional<NamedAttribute> store_set_attr =
      result.attributes.getNamed(CompositeStoreOp::getStoreSetAttrStrName());
  std::optional<NamedAttribute> store_order_attr = result.attributes.getNamed(
      CompositeStoreOp::getStoreOrderMapAttrStrName());
  if (store_set_attr.has_value() == getHaveInputVector ||
      store_order_attr.has_value() == getHaveInputVector) {
    return parser.emitError(parser.getNameLoc())
           << "either store_set/order or input vector has to be present in "
              "composite store";
  }

  if (!getHaveInputVector) {
    auto store_set =
        mlir::dyn_cast<IntegerSetAttr>(store_set_attr->getValue()).getValue();
    // TODO: enhance this with finding constant values and make sure they match.
    if (store_set.getNumDims() < map_attr.getValue().getNumResults()) {
      return parser.emitError(parser.getNameLoc())
             << "Store set and Array dimensions should match";
    }

    auto store_order =
        mlir::dyn_cast<AffineMapAttr>(store_order_attr->getValue()).getValue();
    if (store_set.getNumDims() != store_order.getNumDims()) {
      return parser.emitError(parser.getNameLoc())
             << "Store set and order dimensions should match";
    }
  } else {
    // for coalesce store, store_order attr is not allowed from users. so set
    // store_order to Identity map
    auto num_of_layout_dims = map_attr.getValue().getResults().size();
    auto store_order_map = AffineMap::getMultiDimIdentityMap(
        num_of_layout_dims, builder.getContext());
    result.addAttribute(CompositeStoreOp::getStoreOrderMapAttrStrName(),
                        AffineMapAttr::get(store_order_map));
  }

  return failure(op_result);
}

void CompositeStoreOp::print(OpAsmPrinter& p) {
  auto& op = *this;
  p << ' ' << op.getMemRef() << '[';
  if (AffineMapAttr map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getMapAttrStrName()))
    p.printAffineMapOfSSAIds(map_attr, op.getMapIndices());
  p << ']';
  p.printNewline();
  if (op.getHaveInputVector()) {
    p << "input_vector=";
    p.printOperand(op.getInputVector().value());
  }
  p << " time_symbols(";
  p.printOperands(op.getTimeSymbols());
  p << ')';
  p.printNewline();
  if (op.getHaveInputVector()) {
    p.printOptionalAttrDict(
        op->getAttrs(),
        /*elidedAttrs=*/{
            op.getMapAttrStrName(), op.getNumMemrefIndicesAttrStrName(),
            op.getHaveInputVectorAttrStrName(),
            op.getStoreOrderMapAttrStrName(), op.getStoreSetAttrStrName()});
  } else {
    p.printOptionalAttrDict(
        op->getAttrs(),
        /*elidedAttrs=*/{op.getMapAttrStrName(),
                         op.getNumMemrefIndicesAttrStrName(),
                         op.getHaveInputVectorAttrStrName()});
  }
  p.printNewline();
  if (!op.getHaveInputVector()) {
    p.printRegion(op.getRegion(), false, true);
  }
  p << " : " << op.getMemRefType();
  if (op.getHaveInputVector()) {
    if (auto vtype = dyn_cast<VectorType>(getInputVector().value().getType()))
      p << " , " << vtype;
    else if (auto custom_vtype = dyn_cast<dataflow::CustomVectorType>(
                 getInputVector().value().getType()))
      p << " , " << custom_vtype;
  }
}

void CompositeStoreOp::build(
    OpBuilder& builder, OperationState& result, Value memref,
    /*optional*/ mlir::StringAttr dbgName, AffineMap map, ValueRange operands,
    /*optional*/ IntegerSet store_set,
    /*optional*/ AffineMap store_order, IntegerSet time_set,
    AffineMap time_order, AffineMap time_addr_map, uint32_t num_memref_indices,
    CompositeStoreOp::BodyBuilderFn bodyBuilder) {
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
  result.addOperands(operands);
  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  if (store_set) {
    result.addAttribute(getStoreSetAttrStrName(),
                        IntegerSetAttr::get(store_set));
  }
  result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  if (store_order) {
    result.addAttribute(getStoreOrderMapAttrStrName(),
                        AffineMapAttr::get(store_order));
  }
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(time_addr_map));
  result.addAttribute(getNumMemrefIndicesAttrStrName(),
                      builder.getI32IntegerAttr(num_memref_indices));
  for (auto operand : operands) {
    bool is_vector_type =
        mlir::isa<VectorType, dataflow::CustomVectorType>(operand.getType());
    assert((!is_vector_type) &&
           "Operands of composite_store op can't contain VectorType item for "
           "non-coalesce store");
  }
  result.addAttribute(getHaveInputVectorAttrStrName(),
                      builder.getBoolAttr(false));

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeStoreOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void CompositeStoreOp::build(OpBuilder& builder, OperationState& result,
                             Value memref,
                             /*optional*/ mlir::StringAttr dbgName,
                             AffineMap map, ValueRange operands,
                             IntegerSet time_set, AffineMap time_order,
                             AffineMap time_addr_map,
                             uint32_t num_memref_indices) {
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
  result.addOperands(operands);
  if (dbgName) result.addAttribute(getDbgNameAttrName(result.name), dbgName);
  result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(time_addr_map));
  result.addAttribute(getNumMemrefIndicesAttrStrName(),
                      builder.getI32IntegerAttr(num_memref_indices));
  bool is_vector_type =
      mlir::isa<VectorType, dataflow::CustomVectorType>(operands[0].getType());
  assert((is_vector_type) && "input vector is required for coalesce store");
  result.addAttribute(getHaveInputVectorAttrStrName(),
                      builder.getBoolAttr(true));
  // for coalesce store, store_order attr is not allowed from users. so set
  // store_order to Identity map
  auto num_of_layout_dims = map.getResults().size();
  auto store_order_map = AffineMap::getMultiDimIdentityMap(
      num_of_layout_dims, builder.getContext());
  result.addAttribute(getStoreOrderMapAttrStrName(),
                      AffineMapAttr::get(store_order_map));
  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  CompositeStoreOp::ensureTerminator(*bodyRegion, builder, result.location);
}

CompositeStoreOp CompositeStoreOp::cloneWithNewAccessInfo(
    OpBuilder& builder, const Value& mem_view, AffineMap& subscripts_map,
    const SmallVectorImpl<Value>& indices, const IntegerSet& time_set) {
  // Gather the operands for the new op. These are ordered as follows:
  //   <indices>, <input vector>, <time symbols>
  SmallVector<Value, 16> operands;
  for (auto& index : indices) operands.push_back(index);
  int remaining_operands_idx = getNumMemrefIndices();
  if (getHaveInputVector()) {
    operands.emplace_back(getInputVector().value());
    ++remaining_operands_idx;
  }
  for (std::size_t i = remaining_operands_idx; i < getOperands1().size(); ++i)
    operands.emplace_back(getOperands1()[i]);

  // Create the new comp op in the innermost loop. Builder should already be set
  // to correct location.
  auto& op = *this;
  auto new_comp_store = CompositeStoreOp::create(
      builder, op->getLoc(), mem_view,
      getDbgName().has_value() ? builder.getStringAttr(getDbgName().value())
                               : builder.getStringAttr(""),
      subscripts_map, operands, getStoreSet()->getValue(),
      getStoreOrder().value(), time_set, getTimeOrder(), getTimeAddrMap(),
      indices.size());

  // Delete the yield op automatically inserted to the body. When the original
  // loop body is cloned, the appropriate yield op will also be cloned.
  auto terminator = new_comp_store.getRegion().back().getTerminator();
  if (terminator) terminator->erase();

  // Clone the body.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&new_comp_store.getRegion().front());
  IRMapping ir_map;
  for (auto& o : getRegion().getOps()) (void)builder.clone(*&o, ir_map);
  builder.restoreInsertionPoint(insert_pt);

  return new_comp_store;
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
    op_result =
        op_result || parser.parseColon() ||
        parser.parseOperand(ind_src_memref_info) ||
        parser.parseAffineMapOfSSAIds(
            ind_src_map_operands, ind_src_map_attr,
            CompositeIndirectLoadAndStoreOp::getIndirectSrcMapAttrStrName(),
            result.attributes);
    has_ind_src = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing indirect src";
  op_result = op_result || parser.parseKeyword("direct_src") ||
              parser.parseColon() || parser.parseOperand(dir_src_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_src_map_operands, dir_src_map_attr,
                  CompositeIndirectLoadAndStoreOp::getDirectSrcMapAttrStrName(),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing direct src";
  if (succeeded(parser.parseOptionalKeyword("indirect_dst"))) {
    op_result =
        op_result || parser.parseColon() ||
        parser.parseOperand(ind_dst_memref_info) ||
        parser.parseAffineMapOfSSAIds(
            ind_dst_map_operands, ind_dst_map_attr,
            CompositeIndirectLoadAndStoreOp::getIndirectDstMapAttrStrName(),
            result.attributes);
    has_ind_dst = true;
  }
  if (op_result)
    return parser.emitError(parser.getNameLoc())
           << "error parsing indirect dst";
  op_result = op_result || parser.parseKeyword("direct_dst") ||
              parser.parseColon() || parser.parseOperand(dir_dst_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_dst_map_operands, dir_dst_map_attr,
                  CompositeIndirectLoadAndStoreOp::getDirectDstMapAttrStrName(),
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
      CompositeIndirectLoadAndStoreOp::getLoadSetAttrStrName());
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
      CompositeIndirectLoadAndStoreOp::getLoadOrderMapAttrStrName());
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
      CompositeIndirectLoadAndStoreOp::getStoreSetAttrStrName());
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
      CompositeIndirectLoadAndStoreOp::getStoreOrderMapAttrStrName());
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
    if (AffineMapAttr ind_src_map_attr =
            op->getAttrOfType<AffineMapAttr>(op.getIndirectSrcMapAttrStrName()))
      p.printAffineMapOfSSAIds(ind_src_map_attr, op.getIndirectSrcMapIndices());
    p << ']';
  }

  p << " direct_src:" << op.getDirectSrcMemref() << '[';
  if (AffineMapAttr dir_src_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectSrcMapAttrStrName()))
    p.printAffineMapOfSSAIds(dir_src_map_attr, op.getDirectSrcMapIndices());
  p << ']';

  if (hasIndirectDst()) {
    p << " indirect_dst:" << op.getIndirectDstMemref() << '[';
    if (AffineMapAttr ind_dst_map_attr =
            op->getAttrOfType<AffineMapAttr>(op.getIndirectDstMapAttrStrName()))
      p.printAffineMapOfSSAIds(ind_dst_map_attr, op.getIndirectDstMapIndices());
    p << ']';
  }

  p << " direct_dst:" << op.getDirectDstMemref() << '[';
  if (AffineMapAttr dir_dst_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectDstMapAttrStrName()))
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
          op.getIndirectSrcMapAttrStrName(), op.getDirectSrcMapAttrStrName(),
          op.getIndirectDstMapAttrStrName(), op.getDirectDstMapAttrStrName(),
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
  result.addAttribute(getIndirectSrcMapAttrStrName(),
                      AffineMapAttr::get(indirect_src_map));
  result.addAttribute(getDirectSrcMapAttrStrName(),
                      AffineMapAttr::get(direct_src_map));
  result.addAttribute(getIndirectDstMapAttrStrName(),
                      AffineMapAttr::get(indirect_dst_map));
  result.addAttribute(getDirectDstMapAttrStrName(),
                      AffineMapAttr::get(direct_dst_map));

  result.addAttribute(getLoadSetAttrStrName(), IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderMapAttrStrName(),
                      AffineMapAttr::get(load_order));
  result.addAttribute(getStoreOrderMapAttrStrName(),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrStrName(), IntegerSetAttr::get(store_set));
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getLoadIndirectTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(load_indirect_time_addr_map));
  result.addAttribute(getLoadDirectTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(load_direct_time_addr_map));
  result.addAttribute(getStoreIndirectTimeAddrMapAttrStrName(),
                      AffineMapAttr::get(store_indirect_time_addr_map));
  result.addAttribute(getStoreDirectTimeAddrMapAttrStrName(),
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
      getDbgName().has_value() ? builder.getStringAttr(getDbgName().value())
                               : builder.getStringAttr(""),
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
                  CompositeIndirectLoadOp::getIndirectMapAttrStrName(),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  CompositeIndirectLoadOp::getDirectMapAttrStrName(),
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
      CompositeIndirectLoadOp::getLoadSetAttrStrName());
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
      CompositeIndirectLoadOp::getLoadOrderMapAttrStrName());
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
  if (AffineMapAttr ind_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getIndirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectMapAttrStrName()))
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
      /*elidedAttrs=*/{op.getIndirectMapAttrStrName(),
                       op.getDirectMapAttrStrName(),
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
  result.addAttribute(getIndirectMapAttrStrName(),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrStrName(),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getLoadSetAttrStrName(), IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderMapAttrStrName(),
                      AffineMapAttr::get(load_order));
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrStrName(),
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
                  CompositeIndirectStoreOp::getIndirectMapAttrStrName(),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result = op_result || parser.parseKeyword("direct") ||
              parser.parseColon() || parser.parseOperand(dir_memref_info) ||
              parser.parseAffineMapOfSSAIds(
                  dir_map_operands, dir_map_attr,
                  CompositeIndirectStoreOp::getDirectMapAttrStrName(),
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
      CompositeIndirectStoreOp::getStoreSetAttrStrName());
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
      CompositeIndirectStoreOp::getStoreOrderMapAttrStrName());
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
  if (AffineMapAttr ind_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getIndirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectMapAttrStrName()))
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
      /*elidedAttrs=*/{op.getIndirectMapAttrStrName(),
                       op.getDirectMapAttrStrName(),
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
  result.addAttribute(getIndirectMapAttrStrName(),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrStrName(),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getStoreOrderMapAttrStrName(),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrStrName(), IntegerSetAttr::get(store_set));
  result.addAttribute(getTimeSetAttrStrName(), IntegerSetAttr::get(time_set));
  result.addAttribute(getTimeOrderMapAttrStrName(),
                      AffineMapAttr::get(time_order));
  result.addAttribute(getTimeAddrMapAttrStrName(),
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
  op_result =
      op_result || parser.parseKeyword("indirect") || parser.parseColon() ||
      parser.parseOperand(ind_memref_info) ||
      parser.parseAffineMapOfSSAIds(
          ind_map_operands, ind_map_attr,
          IndirectVectorLoadOp::getIndirectMapAttrStrName(), result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result =
      op_result || parser.parseKeyword("direct") || parser.parseColon() ||
      parser.parseOperand(dir_memref_info) ||
      parser.parseAffineMapOfSSAIds(
          dir_map_operands, dir_map_attr,
          IndirectVectorLoadOp::getDirectMapAttrStrName(), result.attributes);
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

  std::optional<NamedAttribute> load_set_attr =
      result.attributes.getNamed(IndirectVectorLoadOp::getLoadSetAttrStrName());
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
      IndirectVectorLoadOp::getLoadOrderMapAttrStrName());
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
  if (AffineMapAttr ind_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getIndirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  if (hasMulticastInfo()) p << " multicast_info = " << op.getMulticastInfo();

  p.printNewline();
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrStrName(),
                       op.getDirectMapAttrStrName(),
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
  result.addAttribute(getIndirectMapAttrStrName(),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrStrName(),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getLoadSetAttrStrName(), IntegerSetAttr::get(load_set));
  result.addAttribute(getLoadOrderMapAttrStrName(),
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
                  IndirectVectorStoreOp::getIndirectMapAttrStrName(),
                  result.attributes);
  if (op_result)
    return parser.emitError(parser.getNameLoc()) << "error parsing indirect";

  op_result =
      op_result || parser.parseKeyword("direct") || parser.parseColon() ||
      parser.parseOperand(dir_memref_info) ||
      parser.parseAffineMapOfSSAIds(
          dir_map_operands, dir_map_attr,
          IndirectVectorStoreOp::getDirectMapAttrStrName(), result.attributes);
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
      IndirectVectorStoreOp::getStoreSetAttrStrName());
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
      IndirectVectorStoreOp::getStoreOrderMapAttrStrName());
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
  if (AffineMapAttr ind_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getIndirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(ind_map_attr, op.getIndirectMapIndices());
  p << ']';

  p << " direct:" << op.getDirectMemref() << '[';
  if (AffineMapAttr dir_map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getDirectMapAttrStrName()))
    p.printAffineMapOfSSAIds(dir_map_attr, op.getDirectMapIndices());
  p << ']';

  p.printNewline();
  if (hasMulticastInfo()) {
    p << " multicast_info = " << op.getMulticastInfo();
    p.printNewline();
  }
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getIndirectMapAttrStrName(),
                       op.getDirectMapAttrStrName(),
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
  result.addAttribute(getIndirectMapAttrStrName(),
                      AffineMapAttr::get(indirect_map));
  result.addAttribute(getDirectMapAttrStrName(),
                      AffineMapAttr::get(direct_map));

  result.addAttribute(getStoreOrderMapAttrStrName(),
                      AffineMapAttr::get(store_order));
  result.addAttribute(getStoreSetAttrStrName(), IntegerSetAttr::get(store_set));

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
      SymbolicVectorLoadOp::getNumIndicesAttrStrName(), num_indices_attr));

  IntegerAttr num_strides_attr = builder.getI32IntegerAttr(strides.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorLoadOp::getNumStridesAttrStrName(), num_strides_attr));

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
      /*elidedAttrs*/ {op.getNumIndicesAttrStrName(),
                       SymbolicVectorLoadOp::getNumStridesAttrStrName()});
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
      SymbolicVectorStoreOp::getNumIndicesAttrStrName(), num_indices_attr));

  IntegerAttr num_strides_attr = builder.getI32IntegerAttr(strides.size());
  result.attributes.push_back(builder.getNamedAttr(
      SymbolicVectorStoreOp::getNumStridesAttrStrName(), num_strides_attr));

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
      /*elidedAttrs*/ {op.getNumIndicesAttrStrName(),
                       SymbolicVectorStoreOp::getNumStridesAttrStrName()});
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
  auto indices_size = op.getIndices().size();
  if (indices_size < 0) return emitOpError("indices should not be empty");

  // There should be an equal amount of strides to the indices.
  if (op.getStrides().size() != indices_size)
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
  result.addAttribute(getGranularityAttrStrName(),
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
  p << "{ num_slices = " << op.getNumSlices() << " : ";
  p.printType(mlir::cast<IntegerAttr>(op->getAttr(getNumSlicesAttrStrName()))
                  .getType());
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
