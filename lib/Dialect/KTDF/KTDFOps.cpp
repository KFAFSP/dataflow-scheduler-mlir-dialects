//===-- KTDFOps.cpp ---------------------------------------------*- c++ -*-===//
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
// This file implements the ktdf dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
// clang-format on

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDF/KTDF.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom Parsers and Printers
//===----------------------------------------------------------------------===//

namespace {

auto parseCellOrSlot(OpAsmParser& parser,
                     OpAsmParser::UnresolvedOperand& operand,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand>& indices,
                     AffineMapAttr& map,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand>& sizes,
                     DenseI64ArrayAttr& static_sizes) -> ParseResult {
  NamedAttrList attrs;
  if (parser.parseOperand(operand)) {
    return failure();
  }

  // NOTE: We can't use the OptionalSquare delimiter, because the attribute may
  //       not be present if no map was parsed.
  if (!parser.parseOptionalLSquare()) {
    if (parser.parseAffineMapOfSSAIds(indices, map, "map", attrs,
                                      OpAsmParser::Delimiter::None) ||
        parser.parseRSquare()) {
      return failure();
    }
  }

  return failure(!parser.parseOptionalKeyword("size") &&
                 parseDynamicIndexList(parser, sizes, static_sizes));
};

void printCellOrSlot(OpAsmPrinter& printer, Operation* op, Value operand,
                     OperandRange indices, AffineMapAttr map,
                     OperandRange sizes, DenseI64ArrayAttr static_sizes) {
  printer << operand;

  if (isa<MemRefType>(operand.getType())) {
    printer << "[";
    printer.printAffineMapOfSSAIds(map, indices);
    printer << "]";
  }

  if (static_sizes) {
    printer << " size ";
    printDynamicIndexList(printer, op, sizes, static_sizes);
  }
}

template <class Range, class T>
auto allEqualTo(const T& value, Range&& range) -> bool {
  return llvm::all_of(range, [&](const auto&& item) { return value == item; });
}

auto parseBracketPairList(
    OpAsmParser& parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand>& first,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& second) -> ParseResult {
  while (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperand(first.emplace_back()) || parser.parseColon() ||
        parser.parseOperand(second.emplace_back()) || parser.parseRSquare()) {
      return failure();
    }
    std::ignore = parser.parseOptionalComma();
  }

  return success();
}

void printBracketPairList(OpAsmPrinter& printer, Operation* /*op*/,
                          ValueRange first, ValueRange second) {
  llvm::interleaveComma(llvm::zip_equal(first, second), printer,
                        [&](auto pair) {
                          printer << "[" << std::get<0>(pair) << " : "
                                  << std::get<1>(pair) << "]";
                        });
}

}  // namespace

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDF/KTDF.cpp.inc"

//===----------------------------------------------------------------------===//
// TilingDeriveSizeOp
//===----------------------------------------------------------------------===//

auto TilingDeriveSizeOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  SmallVector<OpAsmParser::UnresolvedOperand> ivs;
  SmallVector<OpAsmParser::UnresolvedOperand> tile_sizes;
  OpAsmParser::UnresolvedOperand total_size;
  const auto index_type = parser.getBuilder().getIndexType();

  if (parseBracketPairList(parser, ivs, tile_sizes) ||
      parser.parseKeyword("total_size") || parser.parseEqual() ||
      parser.parseOperand(total_size) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(result.types.emplace_back()) ||
      parser.resolveOperands(ivs, index_type, result.operands) ||
      parser.resolveOperands(tile_sizes, index_type, result.operands) ||
      parser.resolveOperand(total_size, index_type, result.operands)) {
    return failure();
  }

  return success();
}

void TilingDeriveSizeOp::print(OpAsmPrinter& printer) {
  printer << " ";
  printBracketPairList(printer, *this, getIvs(), getTileSizes());
  if (!getIvs().empty()) {
    printer << ", ";
  }
  printer << "total_size = " << getTotalSize();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getType();
}

auto TilingDeriveSizeOp::verify() -> LogicalResult {
  if (getIvs().size() != getTileSizes().size()) {
    return emitOpError("number of ivs must equal number of tile_sizes");
  }
  if (getIvs().empty()) {
    return emitOpError("must have at least one [iv : tile_size] pair");
  }
  return success();
}

auto TilingDeriveSizeOp::fold(FoldAdaptor adaptor) -> OpFoldResult {
  // Only the single-level case is foldable (and is the only case lowered).
  if (getIvs().size() != 1) {
    return {};
  }

  // All three operands must be constant to fold to a constant.
  auto iv_attr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getIvs()[0]);
  auto ts_attr =
      llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getTileSizes()[0]);
  auto total_attr =
      llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getTotalSize());
  if (!iv_attr || !ts_attr || !total_attr) {
    return {};
  }

  // Only fold the single-trip-loop case: a folded enclosing loop replaced its
  // induction variable with the loop's lower bound, 0. For any other constant
  // iv (e.g. an unrolled multi-trip loop) min(tile_size, total_size) would be
  // wrong, so leave the op untouched.
  if (iv_attr.getInt() != 0) {
    return {};
  }

  // Single-trip inner trip count is min(tile_size, total_size).
  const int64_t ts = ts_attr.getInt();
  const int64_t total = total_attr.getInt();
  const int64_t result = ts < total ? ts : total;

  return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), result);
}

//===----------------------------------------------------------------------===//
// TilingLinearizeIndexOp
//===----------------------------------------------------------------------===//

auto TilingLinearizeIndexOp::verify() -> LogicalResult {
  if (getIvs().size() != getStrides().size()) {
    return emitOpError("number of ivs must equal number of strides");
  }
  if (getIvs().empty()) {
    return emitOpError("must have at least one [iv : stride] pair");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PipelineOp
//===----------------------------------------------------------------------===//

auto PipelineOp::getPrivateOp() -> PrivateOp {
  auto ops = getBody()->getOps<PrivateOp>();
  if (ops.begin() == ops.end()) {
    return {};
  }

  return *ops.begin();
}

auto PipelineOp::getStages() -> StageRange {
  return getBody()->getOps<StageOp>();
}

auto PipelineOp::getNumStages() -> unsigned {
  return static_cast<unsigned>(llvm::range_size(getStages()));
}

void PipelineOp::build(OpBuilder& builder, OperationState& state,
                       function_ref<void(OpBuilder&, Location)> body_builder) {
  auto& body = state.addRegion()->emplaceBlock();

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);
    body_builder(builder, state.location);
  }
}

auto PipelineOp::verifyRegions() -> LogicalResult {
  // Verify that all immediate children are StageOp or PrivateOp, and that there
  // is at most one PrivateOp.
  Operation* private_op = nullptr;
  for (auto& op : *getBody()) {
    if (isa<PrivateOp>(op)) {
      if (private_op) {
        auto diag = emitOpError(
            "at most one immediate 'ktdf.private' child is allowed");
        diag.attachNote(private_op->getLoc())
            << "previous 'ktdf.private' is here";
        return diag;
      }
      private_op = &op;
    } else if (!isa<StageOp>(op)) {
      auto diag = emitOpError(
          "immediate children must be 'ktdf.stage' or 'ktdf.private' ops");
      diag.attachNote(op.getLoc()) << "found '" << op.getName() << "' op";
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FifoAllocateOp
//===----------------------------------------------------------------------===//

auto FifoAllocateOp::verify() -> LogicalResult {
  // Count the number of dynamic sizes in the result types
  const auto num_dynamic_sizes =
      static_cast<std::size_t>(llvm::count_if(getSlots(), [](OpResult slot) {
        return cast<FifoSlotType>(slot.getType()).isDynamicNumElements();
      }));

  // Verify that the number of dynamic size operands matches the number of
  // dynamic sizes in the result types
  const auto dynamic_sizes = getDynamicSizes();
  if (dynamic_sizes.size() != num_dynamic_sizes) {
    return emitOpError("number of dynamic size operands (")
           << dynamic_sizes.size()
           << ") must match number of dynamic sizes in result types ("
           << num_dynamic_sizes << ")";
  }

  return success();
}

void FifoAllocateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext*) {
  results.add(
      +[](FifoAllocateOp op, PatternRewriter& rewriter) -> LogicalResult {
        if (op->getUses().empty()) {
          rewriter.eraseOp(op);
        }
        return success();
      });
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

namespace {

// Helper function to check if a value originates from create_token or private
// operation yielding a token
auto isValidTokenSource(Value token_value) -> bool {
  return llvm::TypeSwitch<Operation*, bool>(token_value.getDefiningOp())
      .Case([](CreateTokenOp) { return true; })
      .Case([&](PrivateOp op) {
        const auto result_number =
            cast<OpResult>(token_value).getResultNumber();
        return isValidTokenSource(op.getYieldOp()->getOperand(result_number));
      })
      .Default(false);
}

}  // namespace

auto StageOp::verify() -> LogicalResult {
  // FIXME: Following def-use chains in verifiers is not allowed as per the
  //        MLIR guidelines. This should become a match failure in the affected
  //        passes. In addition, a legalization pass could be added.

  // Verify that all depends_in tokens come from valid sources
  for (auto token : getDependsIn()) {
    if (!isValidTokenSource(token)) {
      return emitOpError(
          "depends_in operand must originate from ktdf.create_token or "
          "ktdf.private operation yielding a token created by "
          "ktdf.create_token");
    }
  }

  // Verify that all depends_out tokens come from valid sources
  for (auto token : getDependsOut()) {
    if (!isValidTokenSource(token)) {
      return emitOpError(
          "depends_out operand must originate from ktdf.create_token or "
          "ktdf.private operation yielding a token created by "
          "ktdf.create_token");
    }
  }

  return success();
}

void StageOp::build(OpBuilder& builder, OperationState& state,
                    ValueRange depends_in, ValueRange depends_out,
                    function_ref<void(OpBuilder&, Location)> body_builder) {
  state.addOperands(depends_in);
  state.addOperands(depends_out);
  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(depends_in.size()),
                                    static_cast<int32_t>(depends_out.size())}));
  auto& body = state.addRegion()->emplaceBlock();

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);
    body_builder(builder, state.location);
  }
}

//===----------------------------------------------------------------------===//
// PrivateOp
//===----------------------------------------------------------------------===//

void PrivateOp::build(OpBuilder& builder, OperationState& state,
                      TypeRange results,
                      function_ref<void(OpBuilder&, Location)> body_builder) {
  state.addTypes(results);
  auto& body = state.addRegion()->emplaceBlock();

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);
    body_builder(builder, state.location);
  }

  // ensureTerminator(*state.regions.front(), builder, state.location);
}

auto PrivateOp::getYieldOp() -> PrivateYieldOp {
  return cast<PrivateYieldOp>(getBody()->getTerminator());
}

namespace {

/// Canonicalizes the results of a PrivateOp.
///
/// - Results without users are dropped.
/// - Values yielded multiple times are coalesced into one result.
/// - External yielded values replace their results.
struct CanonicalizePrivateResults : OpRewritePattern<PrivateOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(PrivateOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!match(op)) {
      return failure();
    }

    PipelinePrivatizer::canonicalize(rewriter, op.getParentOp());
    return success();
  }

 private:
  [[nodiscard]] static auto match(PrivateOp op) -> bool {
    if (llvm::any_of(op.getResults(),
                     [](Value result) -> bool { return result.use_empty(); })) {
      return true;
    }

    auto operands = llvm::to_vector(op.getYieldOp()->getOperands());
    while (!operands.empty()) {
      auto operand = operands.pop_back_val();
      if (operand.getParentRegion()->isAncestor(op->getParentRegion()) ||
          llvm::is_contained(operands, operand)) {
        return true;
      }
    }

    return false;
  }
};

}  // namespace

void PrivateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<CanonicalizePrivateResults>(context);
}

auto PrivateOp::verifyRegions() -> LogicalResult {
  // Verify that the number and types of yield operands match the results
  const auto yield_operands = getBody()->getTerminator()->getOperands();
  const auto results = getResults();

  if (yield_operands.size() != results.size()) {
    return emitOpError("yield operands count (")
           << yield_operands.size() << ") must match results count ("
           << results.size() << ")";
  }

  for (auto [yield_operand, result] : llvm::zip(yield_operands, results)) {
    if (yield_operand.getType() != result.getType()) {
      return emitOpError("yield operand type ")
             << yield_operand.getType() << " must match result type "
             << result.getType();
    }
  }

  return success();
}

void PrivateOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor>& regions) {
  if (!point.isParent()) {
    regions.emplace_back(getOperation(), getResults());
    return;
  }

  regions.emplace_back(&getBodyRegion());
}

//===----------------------------------------------------------------------===//
// DataTransferOp
//===----------------------------------------------------------------------===//

auto DataTransferOp::verify() -> LogicalResult {
  const auto verify_cell = [&](Twine side_name, MemRefType type, AffineMap map,
                               OperandRange indices,
                               ArrayRef<int64_t> sizes) -> LogicalResult {
    if (map.getNumSymbols() != 0) {
      return emitOpError(side_name)
             << "_map must have 0 symbols (dims-only); got "
             << map.getNumSymbols();
    }
    if (map.getNumResults() != static_cast<unsigned>(type.getRank())) {
      return emitOpError(side_name)
             << " memref has rank " << type.getRank() << " but " << side_name
             << "_map has " << map.getNumResults() << " results";
    }
    if (indices.size() != map.getNumDims()) {
      return emitOpError(side_name)
             << "_indices count (" << indices.size() << ") must match "
             << side_name << "_map dim count (" << map.getNumDims() << ")";
    }
    if (sizes.size() != static_cast<unsigned>(type.getRank())) {
      return emitOpError(side_name)
             << " size count (" << sizes.size() << ") must match memref rank ("
             << type.getRank() << ")";
    }
    return success();
  };

  const auto verify_side =
      [&](Twine side_name, Type type, std::optional<AffineMap> map,
          OperandRange indices, ArrayRef<int64_t> sizes) -> LogicalResult {
    return llvm::TypeSwitch<Type, LogicalResult>(type)
        .Case([&](MemRefType memref_type) -> LogicalResult {
          if (!map) {
            return emitOpError(side_name) << "_map must be present when "
                                          << side_name << " is a memref";
          }
          return verify_cell(side_name, memref_type, *map, indices, sizes);
        })
        .Case([&](FifoSlotType) -> LogicalResult {
          if (map) {
            return emitOpError(side_name)
                   << "_map must be absent for fifo " << side_name;
          }
          if (!indices.empty()) {
            return emitOpError("FIFO slot ")
                   << side_name << " cannot have indices specified";
          }
          return success();
        });
  };

  if (failed(verify_side(
          "source", getSource().getType(), getSourceMap(), getSourceIndices(),
          getStaticSourceSizes().value_or(ArrayRef<int64_t>{}))) ||
      failed(verify_side("dest", getDestination().getType(), getDestMap(),
                         getDestIndices(),
                         getStaticDestSizes().value_or(ArrayRef<int64_t>{})))) {
    return failure();
  }

  // Sizes consistency (dynamic operand count vs static array placeholders).
  const auto verify_size =
      [&](Twine side_name, OperandRange dynamic_sizes,
          std::optional<ArrayRef<int64_t>> static_sizes) -> LogicalResult {
    if (!static_sizes) {
      return success();
    }
    const unsigned num_dynamic = llvm::count_if(
        *static_sizes, [](auto val) { return ShapedType::isDynamic(val); });
    if (dynamic_sizes.size() != num_dynamic) {
      return emitOpError("number of dynamic ")
             << side_name << " sizes (" << dynamic_sizes.size()
             << ") must match number of dynamic entries in static_" << side_name
             << "_sizes (" << num_dynamic << ")";
    }
    return success();
  };

  if (failed(verify_size("source", getSourceSizes(), getStaticSourceSizes())) ||
      failed(verify_size("dest", getDestSizes(), getStaticDestSizes()))) {
    return failure();
  }

  return success();
}

auto DataTransferOp::getAffineMapAttrForMemRef(Value memref) -> NamedAttribute {
  if (memref == getSource()) {
    return {getSourceMapAttrName(), getSourceMapAttr()};
  }
  assert(memref == getDestination() &&
         "memref must be either source or destination of this op");
  return {getDestMapAttrName(), getDestMapAttr()};
}

// Build with mixed static/dynamic sizes (OpFoldResult).
// Caller is responsible for passing AffineMap() (null) for fifo sides and
// a real AffineMap for memref sides; the verifier enforces this invariant
// on the final IR. The builder does not assert on construction because
// transient IR may legitimately reference access-tile-typed values that
// will be lowered to memrefs later in the pipeline.
void DataTransferOp::build(OpBuilder& builder, OperationState& state,
                           Value source, AffineMap source_map,
                           ValueRange source_indices,
                           ArrayRef<OpFoldResult> source_sizes,
                           Value destination, AffineMap dest_map,
                           ValueRange dest_indices,
                           ArrayRef<OpFoldResult> dest_sizes) {
  SmallVector<Value> dynamic_source_sizes;
  SmallVector<int64_t> static_source_sizes;
  dispatchIndexOpFoldResults(source_sizes, dynamic_source_sizes,
                             static_source_sizes);

  SmallVector<Value> dynamic_dest_sizes;
  SmallVector<int64_t> static_dest_sizes;
  dispatchIndexOpFoldResults(dest_sizes, dynamic_dest_sizes, static_dest_sizes);

  build(builder, state, source, source_indices, dynamic_source_sizes,
        destination, dest_indices, dynamic_dest_sizes,
        builder.getDenseI64ArrayAttr(static_source_sizes),
        builder.getDenseI64ArrayAttr(static_dest_sizes),
        source_map ? AffineMapAttr::get(source_map) : AffineMapAttr(),
        dest_map ? AffineMapAttr::get(dest_map) : AffineMapAttr());
}

// Build with all static sizes.
void DataTransferOp::build(OpBuilder& builder, OperationState& state,
                           Value source, AffineMap source_map,
                           ValueRange source_indices,
                           ArrayRef<int64_t> source_sizes, Value destination,
                           AffineMap dest_map, ValueRange dest_indices,
                           ArrayRef<int64_t> dest_sizes) {
  const auto source_size_values =
      llvm::map_to_vector(source_sizes, [&](int64_t value) -> OpFoldResult {
        return builder.getI64IntegerAttr(value);
      });
  const auto dest_size_values =
      llvm::map_to_vector(dest_sizes, [&](int64_t value) -> OpFoldResult {
        return builder.getI64IntegerAttr(value);
      });

  build(builder, state, source, source_map, source_indices, source_size_values,
        destination, dest_map, dest_indices, dest_size_values);
}

void DataTransferOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  if (isSourceFifo()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                         FifoResource::get());

    // Reading from a FIFO mutates its state, and thus it is also writing. But
    // since we don't know the other FIFO slots, we have to pessimistcally
    // clobber everything.
    effects.emplace_back(MemoryEffects::Write::get(), FifoResource::get());
  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  }

  if (isDestFifo()) {
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable(),
                         0, false, FifoResource::get());
  } else {
    effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
  }
}

//===----------------------------------------------------------------------===//
// ReadFromFifoOp
//===----------------------------------------------------------------------===//

namespace {

auto verifyFifoReadWrite(FifoSlotType slot, ShapedType shaped,
                         function_ref<InFlightDiagnostic()> emit_error)
    -> LogicalResult {
  const auto fifo_elements = slot.getNumElements();

  // Both must be dynamic or both must be static with matching values
  const auto shaped_dynamic = shaped.getNumDynamicDims() != 0;
  if (ShapedType::isDynamic(fifo_elements) != shaped_dynamic) {
    return emit_error()
           << "FIFO slot and data must both be dynamic or both be static";
  }

  // Verify total number of elements matches FIFO slot size
  const auto shaped_elements = shaped.getNumElements();
  if (!shaped_dynamic && shaped_elements != fifo_elements) {
    return emit_error() << "data total elements (" << shaped_elements
                        << ") must match FIFO slot size (" << fifo_elements
                        << ")";
  }

  return success();
}

}  // namespace

auto ReadFromFifoOp::verify() -> LogicalResult {
  return verifyFifoReadWrite(getFifoSlot().getType(),
                             llvm::cast<ShapedType>(getResult().getType()),
                             [&]() { return emitOpError(); });
}

void ReadFromFifoOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getFifoSlotMutable(),
                       FifoResource::get());

  // Reading from a FIFO mutates its state, and thus it is also writing. But
  // since we don't know the other FIFO slots, we have to pessimistcally
  // clobber everything.
  effects.emplace_back(MemoryEffects::Write::get(), FifoResource::get());

  if (isa<MemRefType>(getResult().getType())) {
    // The returned memref is a transient buffer that we allocate and populate.
    effects.emplace_back(MemoryEffects::Allocate::get(),
                         cast<OpResult>(getResult()), 0, true,
                         SideEffects::AutomaticAllocationScopeResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         cast<OpResult>(getResult()));
  }
}

//===----------------------------------------------------------------------===//
// WriteToFifoOp
//===----------------------------------------------------------------------===//

auto WriteToFifoOp::verify() -> LogicalResult {
  return verifyFifoReadWrite(getFifoSlot().getType(),
                             llvm::cast<ShapedType>(getData().getType()),
                             [&]() { return emitOpError(); });
}

void WriteToFifoOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  if (isa<MemRefType>(getData().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDataMutable(), 0,
                         false, SideEffects::DefaultResource::get());
  }

  effects.emplace_back(MemoryEffects::Write::get(), &getFifoSlotMutable(), 0,
                       false, FifoResource::get());
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(
    OpBuilder& builder, OperationState& state, ValueRange lower_bounds,
    ValueRange upper_bounds, ValueRange steps, int64_t num_instances,
    function_ref<void(OpBuilder&, Location, ValueRange, Value)> body_builder) {
  state.addOperands(lower_bounds);
  state.addOperands(upper_bounds);
  state.addOperands(steps);
  state.addAttribute(getNumInstancesAttrName(state.name),
                     builder.getI64IntegerAttr(num_instances));

  // Create the body region with a single block. The block has one index
  // argument per induction variable plus one trailing instance-id argument.
  auto& body = state.addRegion()->emplaceBlock();
  const auto num_loops = lower_bounds.size();
  SmallVector<Type> arg_types(num_loops + 1U, builder.getIndexType());
  SmallVector<Location> arg_locs(num_loops + 1U, state.location);
  body.addArguments(arg_types, arg_locs);

  if (body_builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);
    body_builder(builder, state.location, body.getArguments().drop_back(),
                 body.getArguments().back());
  }
}

auto ParallelOp::parse(OpAsmParser& parser, OperationState& result)
    -> ParseResult {
  auto& builder = parser.getBuilder();
  const auto index_ty = builder.getIndexType();

  // Parse the block-argument list: `(` %iv1, %iv2, ..., %inst `)`.
  SmallVector<OpAsmParser::Argument> body_args;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
        auto& arg = body_args.emplace_back();
        arg.type = index_ty;
        return parser.parseArgument(arg);
      })) {
    return failure();
  }

  if (body_args.size() < 2) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least one induction variable plus an "
                            "instance id (got ")
           << body_args.size() << " block arguments)";
  }

  const auto num_loops = body_args.size() - 1U;

  SmallVector<OpAsmParser::UnresolvedOperand> operands;

  // Parse `=` `(` lower_bounds `)` `to` `(` upper_bounds `)` `step` `(` steps
  // `)`.
  if (parser.parseEqual() ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren, true,
                              num_loops) ||
      parser.parseKeyword("to") ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren, true,
                              2 * num_loops) ||
      parser.parseKeyword("step") ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren, true,
                              3 * num_loops)) {
    return failure();
  }

  // Parse `distribute` `(` `num_instances` `=` integer-literal `)`.
  IntegerAttr num_instances_attr;
  if (parser.parseKeyword("distribute") || parser.parseLParen() ||
      parser.parseKeyword("num_instances") || parser.parseEqual() ||
      parser.parseAttribute(num_instances_attr, builder.getI64Type(),
                            ParallelOp::getNumInstancesAttrName(result.name),
                            result.attributes) ||
      parser.parseRParen()) {
    return failure();
  }

  // Resolve operands.
  if (parser.resolveOperands(operands, index_ty, result.operands)) {
    return failure();
  }

  // Parse the region using the named block arguments.
  Region* body = result.addRegion();
  if (parser.parseRegion(*body, body_args)) {
    return failure();
  }

  // Optional trailing attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ParallelOp::print(OpAsmPrinter& printer) {
  // Print block-argument list: `(` %iv1, ..., %inst `)`. Block args are all
  // index-typed by construction (see verify), so we only print the SSA names
  // here — the parser binds them as index without type annotations, matching
  // the spec's surface form.
  printer << " (";
  llvm::interleaveComma(getBody()->getArguments(), printer,
                        [&](BlockArgument arg) { printer.printOperand(arg); });
  printer << ")";

  // Print `= (lbs) to (ubs) step (steps)`.
  printer << " = (";
  printer.printOperands(getLowerBounds());
  printer << ") to (";
  printer.printOperands(getUpperBounds());
  printer << ") step (";
  printer.printOperands(getSteps());
  printer << ")";

  // Print `distribute(num_instances = N)`.
  printer << " distribute(num_instances = " << getNumInstances() << ")";

  // Print the region.
  printer << " ";
  printer.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  // Print remaining attributes (suppress the ones already printed inline).
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getNumInstancesAttrName()});
}

auto ParallelOp::verify() -> LogicalResult {
  const auto num_loops = getNumLoops();

  if (num_loops < 1) {
    return emitOpError("expected at least one induction variable");
  }

  if (getNumInstances() < 1) {
    return emitOpError("expected num_instances >= 1, got ")
           << getNumInstances();
  }

  auto& body = *getBody();
  const auto expected_args = num_loops + 1;
  if (body.getNumArguments() != expected_args) {
    return emitOpError("expected ")
           << expected_args
           << " block arguments (one per induction variable plus the instance "
              "id), got "
           << body.getNumArguments();
  }
  for (BlockArgument arg : body.getArguments()) {
    if (!arg.getType().isIndex()) {
      return emitOpError(
          "all block arguments (induction variables and instance id) must be "
          "of type index");
    }
  }

  return success();
}

auto ParallelOp::verifyRegions() -> LogicalResult {
  // The custom parser's parseRegion already enforces a terminator, so this
  // check fires only when the op is constructed via the generic op form or
  // programmatically without a terminator.
  auto& body = *getBody();
  if (body.empty() || !isa<ParallelYieldOp>(body.back())) {
    return emitOpError(
        "region must be terminated by a ktdf.parallel_yield operation");
  }

  return success();
}

auto ParallelOp::getLoopRegions() -> SmallVector<Region*> {
  return {&getBodyRegion()};
}

auto ParallelOp::getLoopInductionVars() -> std::optional<SmallVector<Value>> {
  return llvm::to_vector(ValueRange(getBody()->getArguments()));
}

auto ParallelOp::getLoopLowerBounds()
    -> std::optional<SmallVector<OpFoldResult>> {
  return getAsOpFoldResult(getLowerBounds());
}

auto ParallelOp::getLoopSteps() -> std::optional<SmallVector<OpFoldResult>> {
  return getAsOpFoldResult(getSteps());
}

auto ParallelOp::getLoopUpperBounds()
    -> std::optional<SmallVector<OpFoldResult>> {
  return getAsOpFoldResult(getUpperBounds());
}

namespace {

[[nodiscard]] auto mpMul(const APInt& lhs, const APInt& rhs) -> APInt {
  const auto width = std::max(1U, lhs.getActiveBits() + rhs.getActiveBits());
  return lhs.zext(width) * rhs.zext(width);
}

}  // namespace

auto ParallelOp::getStaticTripCount() -> std::optional<APInt> {
  APInt result(64U, 1);

  for (auto [lb, ub, step] :
       llvm::zip_equal(getLowerBounds(), getUpperBounds(), getSteps())) {
    const auto slice =
        mlir::constantTripCount(lb, ub, step, false, scf::computeUbMinusLb);
    if (!slice) {
      return std::nullopt;
    }

    result = mpMul(result, *slice);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// BufferPhaseOp
//===----------------------------------------------------------------------===//

auto BufferPhaseOp::verify() -> LogicalResult {
  if (getIvs().empty()) {
    return emitOpError("requires at least one induction variable operand");
  }
  // NOTE: We intentionally do NOT verify that each operand is an scf.for
  // induction variable. Canonicalization can legitimately fold a single-trip
  // enclosing loop away, replacing its IV operand with a constant; the
  // BufferPhaseOp folder drops such constant operands. Enforcing the IV
  // invariant here (by walking def-use chains) both violates MLIR verifier
  // guidelines and rejects valid intermediate IR. The consuming lowering pass
  // (BufferPhaseLowering) remains the enforcement point for the IV invariant.
  return success();
}

auto BufferPhaseOp::fold(FoldAdaptor adaptor) -> OpFoldResult {
  // Partition operands into constants (folded-away single-trip loops) and
  // non-constants (live IVs). adaptor.getIvs() yields a constant Attribute for
  // each operand that is a constant, or null otherwise.
  llvm::SmallVector<mlir::Value> kept;
  for (auto [value, attr] : llvm::zip(getIvs(), adaptor.getIvs())) {
    if (!attr) {
      kept.push_back(value);
    }
  }

  // No constants: nothing to fold.
  if (kept.size() == getIvs().size()) {
    return {};
  }

  // All operands constant: the phase no longer varies, fold to index 0.
  if (kept.empty()) {
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 0);
  }

  // Some constants: drop them in place, keep the live IVs. Returning the
  // op's own result signals an in-place operand mutation to the folder driver.
  getIvsMutable().assign(kept);
  return getResult();
}

//===----------------------------------------------------------------------===//
// SelectMemrefOp
//===----------------------------------------------------------------------===//

auto SelectMemrefOp::verify() -> LogicalResult {
  const auto candidates = getCandidates();
  if (candidates.size() < 2) {
    return emitOpError("requires at least 2 candidate memrefs, got ")
           << candidates.size();
  }

  // NOTE: We intentionally do NOT verify that the phase operand originates from
  // a ktdf.buffer_phase op. Canonicalization (e.g. BufferPhaseOp::fold) can
  // legitimately replace the phase with a constant; enforcing the source-op
  // invariant here rejects valid intermediate IR. The consuming lowering pass
  // (BufferPhaseLowering) remains the enforcement point for that invariant.
  // The phase operand must come from a ktdf.buffer_phase whose num_phases
  // equals the number of candidates.
  auto phase_op = getPhase().getDefiningOp<BufferPhaseOp>();
  if (phase_op && phase_op.getNumPhases() != candidates.size()) {
    return emitOpError("phase op num_phases (")
           << phase_op.getNumPhases() << ") must equal candidate count ("
           << candidates.size() << ")";
  }

  return success();
}

auto SelectMemrefOp::fold(FoldAdaptor adaptor) -> OpFoldResult {
  // A constant phase selects a fixed candidate.
  auto phase_attr =
      llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getPhase());
  if (!phase_attr) {
    return {};
  }
  const int64_t phase = phase_attr.getInt();
  if (phase < 0 || phase >= static_cast<int64_t>(getCandidates().size())) {
    return {};
  }
  return getCandidates()[phase];
}

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

void OpaqueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  for (auto& input : getInputsMutable()) {
    if (!isa<MemRefType>(input.get().getType())) {
      continue;
    }

    effects.emplace_back(MemoryEffects::Read::get(), &input);
  }

  for (auto& output : getOutputsMutable()) {
    if (!isa<MemRefType>(output.get().getType())) {
      continue;
    }

    effects.emplace_back(MemoryEffects::Read::get(), &output);
    effects.emplace_back(MemoryEffects::Write::get(), &output);
  }
}

auto OpaqueOp::getSpeculatability() -> Speculation::Speculatability {
  return hasPureTensorSemantics() ? Speculation::Speculatable
                                  : Speculation::NotSpeculatable;
}
