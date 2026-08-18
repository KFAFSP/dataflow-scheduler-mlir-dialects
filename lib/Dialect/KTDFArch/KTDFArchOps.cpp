//===-- KTDFArchOps.cpp -----------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDL/IR/PDLOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/WalkResult.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// Custom Traits
//===----------------------------------------------------------------------===//

auto mlir::ktdf_arch::verifySubgraph(Operation* op) -> LogicalResult {
  for (auto& region : op->getRegions()) {
    for (auto& child : region.getOps()) {
      if (child.hasTrait<OpTrait::IsTerminator>() || child.hasTrait<IsMeta>()) {
        continue;
      }

      if (!isa<Resource>(child)) {
        auto diag = op->emitOpError("expects child ops to be resources");
        diag.attachNote(child.getLoc()) << "unexpected child is here";
        return diag;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Custom Assembly Format Declarations
//===----------------------------------------------------------------------===//

namespace {

auto parseShortTypeList(OpAsmParser& parser, SmallVectorImpl<Type>& types)
    -> ParseResult {
  return parser.parseCommaSeparatedList([&]() -> ParseResult {
    return parseShortType(parser, types.emplace_back());
  });
}

void printShortTypeList(OpAsmPrinter& printer, Operation* op, TypeRange types) {
  llvm::interleaveComma(types, printer,
                        [&](Type type) { printShortType(printer, op, type); });
}

auto parseAttrDictOrAlias(OpAsmParser& parser, NamedAttrList& attrs)
    -> ParseResult {
  DictionaryAttr attr;
  if (const auto maybe_result = parser.parseOptionalAttribute(attr);
      maybe_result.has_value()) {
    if (maybe_result.value()) {
      return failure();
    }

    attrs.append(attr.getValue());
  }

  return success();
}

void printAttrDictOrAlias(OpAsmPrinter& printer, Operation* op,
                          const NamedAttrList& attrs,
                          ArrayRef<StringRef> elided_names = {}) {
  DictionaryAttr dict;
  if (elided_names.empty()) {
    if (attrs.empty()) {
      return;
    }
    dict = DictionaryAttr::get(op->getContext(), attrs);
  } else {
    NamedAttrList copy(attrs);
    for (auto name : elided_names) {
      copy.erase(name);
    }
    if (copy.empty()) {
      return;
    }
    dict = DictionaryAttr::get(op->getContext(), copy);
  }

  printer << " " << dict;
}

auto parseGraphOrImport(OpAsmParser& parser, Region& region,
                        StringAttr& import_path) -> ParseResult {
  import_path = {};
  if (const auto maybe_result = parser.parseOptionalRegion(region);
      maybe_result.has_value()) {
    return maybe_result.value();
  }

  if (parser.parseKeyword("import") || parser.parseLParen() ||
      parser.parseAttribute(import_path) || parser.parseRParen()) {
    return failure();
  }

  return success();
}

void printGraphOrImport(OpAsmPrinter& printer, Operation* /*op*/,
                        Region& region, StringAttr import_path) {
  if (import_path) {
    printer << "import(" << import_path << ")";
    return;
  }

  printer.printRegion(region, false, false);
}

auto parseSwitchType(OpAsmParser& parser, SmallVectorImpl<Type>& types)
    -> ParseResult {
  unsigned num_ports;
  if (parser.parseLSquare() || parser.parseInteger(num_ports) ||
      parser.parseRSquare()) {
    return failure();
  }

  types.resize(num_ports, PortType::get(parser.getContext()));
  return success();
}

void printSwitchType(OpAsmPrinter& printer, Operation* /*op*/,
                     TypeRange types) {
  printer << "[" << types.size() << "]";
}

}  // namespace

//===----------------------------------------------------------------------===//
// KTDFArchDialect
//===----------------------------------------------------------------------===//

void KTDFArchDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.cpp.inc"

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//

void DeviceOp::build(OpBuilder& /*builder*/, OperationState& state,
                     StringAttr sym_name, StringAttr import_path) {
  state.addAttribute(getSymNameAttrName(state.name), sym_name);

  if (!import_path || import_path.empty()) {
    state.addRegion()->emplaceBlock();
  } else {
    state.addRegion();
    state.addAttribute(getImportPathAttrName(state.name), import_path);
  }
}

auto DeviceOp::verify() -> LogicalResult {
  if (getRegion().empty()) {
    if (!getImportPath()) {
      return emitOpError("empty device requires '")
             << getImportPathAttrName().strref() << "' attribute";
    }

    return success();
  }

  if (getImportPath()) {
    return emitOpError("non-empty device can't have '")
           << getImportPathAttrName().strref() << "' attribute";
  }

  return success();
}

auto DeviceOp::verifyRegions() -> LogicalResult {
  llvm::StringMap<Operation*> ids;

  const auto visit = [&](Operation* op) -> WalkResult {
    auto resource = dyn_cast<Resource>(op);
    if (!resource) {
      return WalkResult::advance();
    }

    if (const auto maybe_id = resource.getId(); maybe_id) {
      const auto [it, added] = ids.try_emplace(*maybe_id, op);
      if (!added) {
        auto diag = op->emitError("resource with id \"")
                    << *maybe_id << "\" redefined";
        diag.attachNote(it->second->getLoc()) << "previous definition is here";
        return WalkResult::interrupt();
      }
    }

    if (!op->hasTrait<IsSubgraph>()) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  };

  return success(!walk(visit).wasInterrupted());
}

//===----------------------------------------------------------------------===//
// GroupOp
//===----------------------------------------------------------------------===//

auto GroupOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  // [ symbol-name ]
  {
    StringAttr sym_name;
    std::ignore = parser.parseOptionalSymbolName(
        sym_name, getIdAttrName(result.name), result.attributes);
  }

  // dictionary-attr
  if (parseAttrDictOrAlias(parser, result.attributes)) {
    return failure();
  }

  // `share` `(` [ ssa-name { `,` ssa-name } ] `)`
  SmallVector<OpAsmParser::Argument> capture_args;
  if (parser.parseKeyword("share") ||
      parser.parseArgumentList(capture_args, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  const auto memory_type = MemoryType::get(parser.getContext());
  for (auto& arg : capture_args) {
    arg.type = memory_type;
    if (parser.resolveOperand(arg.ssaName, arg.type, result.operands)) {
      return failure();
    }
  }

  // region
  auto& region = *result.addRegion();
  if (parser.parseRegion(region, capture_args, true)) {
    return failure();
  }
  ensureTerminator(region, parser.getBuilder(), result.location);

  // [ `->` short-type-list ]
  if (!parser.parseOptionalArrow() &&
      parseShortTypeList(parser, result.types)) {
    return failure();
  }

  return success();
}

void GroupOp::print(OpAsmPrinter& printer) {
  // [ symbol-name ]
  if (const auto id = getIdAttr(); id) {
    printer << " ";
    printer.printSymbolName(id);
  }

  // [ dictionary-attr ]
  printAttrDictOrAlias(printer, *this, (*this)->getAttrs(), {getIdAttrName()});

  // `share` `(` [ ssa-name [ attr-dict ] { `,` ssa-name [ attr-dict ] } ] `)`
  printer.shadowRegionArgs(getRegion(), getOperands());
  printer << " share(";
  printer.printOperands(getOperands());
  printer << ") ";

  // region
  printer.printRegion(getRegion(), false,
                      getBody()->getTerminator()->getNumOperands() > 0);

  // [ `->` short-type-list ]
  if (const auto result_types = getResultTypes(); !result_types.empty()) {
    printer << " -> ";
    printShortTypeList(printer, *this, result_types);
  }
}

void GroupOp::build(
    OpBuilder& builder, OperationState& state, ValueRange shared_memory,
    unsigned num_results,
    function_ref<void(OpBuilder&, Location, ValueRange)> body_builder) {
  state.addOperands(shared_memory);
  state.types.resize(num_results, ExecutionUnitType::get(builder.getContext()));

  auto& body = state.addRegion()->emplaceBlock();
  SmallVector<Location> arg_locs(shared_memory.size(), state.location);
  body.addArguments(shared_memory, arg_locs);

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);
    body_builder(builder, state.location, body.getArguments());
  }
}

auto GroupOp::getEntrySuccessorOperands(RegionSuccessor successor)
    -> OperandRange {
  if (successor.isParent()) {
    return getBody()->getTerminator()->getOperands();
  }

  return getOperands();
}

void GroupOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor>& regions) {
  if (point.isParent()) {
    regions.emplace_back(&getRegion(), getRegion().getArguments());
    return;
  }

  regions.emplace_back(getOperation(), getResults());
}

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

namespace {

[[nodiscard]] auto isMemorySpaceAttr(Attribute attr) -> bool {
  // See mlir::detail::isSupportedMemorySpace(Attribute) -> bool
  return attr && (!isa<BuiltinDialect>(attr.getDialect()) ||
                  isa<IntegerAttr, StringAttr, DictionaryAttr>(attr));
}

}  // namespace

auto MemoryOp::verify() -> LogicalResult {
  if (!isMemorySpaceAttr(getKind())) {
    return emitOpError("attribute 'kind' requires valid memory space");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

auto SwitchOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  if (parseSwitchType(parser, result.types)) {
    return failure();
  }

  if (parseAttrDictOrAlias(parser, result.attributes)) {
    return failure();
  }

  const auto connectivity_name = getConnectivityAttrName(result.name);
  if (!result.attributes.get(connectivity_name)) {
    const auto num_ports = static_cast<int64_t>(result.types.size());
    const auto type = RankedTensorType::get({num_ports, num_ports},
                                            parser.getBuilder().getI1Type());
    result.addAttribute(connectivity_name,
                        DenseIntElementsAttr::get(type, true));
  }

  return success();
}

void SwitchOp::print(OpAsmPrinter& printer) {
  printer << " ";

  printSwitchType(printer, *this, getResultTypes());

  SmallVector<StringRef> elided;
  if (getConnectivity().isSplat() && getConnectivity().getSplatValue<bool>()) {
    elided.push_back(getConnectivityAttrName());
  }

  printAttrDictOrAlias(printer, *this, (*this)->getAttrs(), elided);
}

void SwitchOp::build(OpBuilder& builder, OperationState& state,
                     unsigned num_ports, ElementsAttr connectivity) {
  state.types.resize(num_ports, PortType::get(builder.getContext()));
  if (!connectivity) {
    const auto type =
        RankedTensorType::get({num_ports, num_ports}, builder.getI1Type());
    connectivity = DenseIntElementsAttr::get(type, true);
  }

  state.addAttribute(getConnectivityAttrName(state.name), connectivity);
}

auto SwitchOp::verify() -> LogicalResult {
  const auto num_ports = static_cast<int64_t>(getNumResults());
  const auto connectivity_shape = getConnectivity().getShapedType().getShape();
  if (connectivity_shape != ArrayRef({num_ports, num_ports})) {
    auto diag = emitOpError("expected connectivity shape {")
                << num_ports << ", " << num_ports << "}, but got {";
    return diag;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DatapathOp
//===----------------------------------------------------------------------===//

auto DatapathOp::verify() -> LogicalResult {
  if (getSource() == getTarget()) {
    return emitOpError("can't link resource to itself");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PatternsOp
//===----------------------------------------------------------------------===//

void PatternsOp::build(OpBuilder& builder, OperationState& result,
                       ArrayRef<StringRef> groups,
                       function_ref<void(OpBuilder&, Location)> body_builder) {
  auto& props = result.getOrAddProperties<Properties>();
  props.groups = builder.getStrArrayAttr(groups);

  auto& body = result.addRegion()->emplaceBlock();

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPointToStart(&body);
    body_builder(builder, result.location);
  }
}

auto PatternsOp::verifyRegions() -> LogicalResult {
  for (auto& child : getOps()) {
    if (!isa<pdl::PatternOp>(child)) {
      auto diag = emitOpError("expects child ops to be '")
                  << pdl::PatternOp::getOperationName() << "'";
      diag.attachNote(child.getLoc()) << "unexpected child is here";
      return diag;
    }
  }

  return success();
}
