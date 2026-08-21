//===-- KTDFArchIntrinsics.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/TypeName.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

auto emitIntrinsicError(Operation* op, const NamedAttribute& intrinsic)
    -> InFlightDiagnostic {
  return op->emitError("'") << intrinsic.getName().strref() << "' intrisic ";
}

template <class T, class = void>
struct has_op_name : std::false_type {};
template <class T>
struct has_op_name<T, std::void_t<decltype(T::getOperationName())>>
    : std::true_type {};
template <class T>
inline constexpr auto has_op_name_v = has_op_name<T>::value;

template <class T>
[[nodiscard]] auto getOpName() -> StringRef {
  if constexpr (has_op_name_v<T>) {
    const auto name = T::getOperationName();
    const auto sep = name.find('.');
    if (name.substr(0, sep) == KTDFArchDialect::getDialectNamespace()) {
      return name.substr(sep + 1);
    }

    return name;
  } else {
    const auto name = llvm::getTypeName<T>();
    const auto sep = name.find_last_of(':');
    return name.substr(sep != StringRef::npos ? sep + 1 : 0);
  }
}

template <class... Nodes>
auto verifyOn(EmitErrorFn emit_error, Operation* op) -> LogicalResult {
  if (!isa<Nodes...>(op)) {
    auto diag = emit_error() << "only valid on ";
    llvm::interleaveComma(ArrayRef<StringRef>{getOpName<Nodes>()...}, diag);
    return failure();
  }

  return success();
}

template <class Feature, class... Nodes>
auto verifyFeature(Operation* op, const NamedAttribute& attr) -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (isa<KTDFArchDialect>(op->getDialect()) &&
      failed(verifyOn<Nodes...>(emit_error, op))) {
    return failure();
  }

  const auto value = dyn_cast<Feature>(attr.getValue());
  if (!value) {
    return emit_error() << "requires unit or dictionary attribute";
  }
  return value.verify(emit_error);
}

}  // namespace

//===----------------------------------------------------------------------===//
// BandwidthAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyBandwidthAttr(Operation* op,
                                          const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (failed(verifyOn<Link>(emit_error, op))) {
    return failure();
  }

  const auto value = dyn_cast<BandwidthAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires 64-bit integer attribute";
  }
  return value.verify(emit_error);
}

auto BandwidthAttr::get(MLIRContext* context, int64_t value) -> BandwidthAttr {
  return cast<BandwidthAttr>(I64Attr::get(context, value));
}

auto BandwidthAttr::verify(EmitErrorFn emit_error) const -> LogicalResult {
  if (getValue() <= 0) {
    return emit_error() << "value must be > 0";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FeaturesAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeaturesAttr(Operation* op,
                                         const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  const auto value = dyn_cast<FeaturesAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires dictionary attribute";
  }
  return value.verify(op);
}

auto FeaturesAttr::verify(Operation* op) const -> LogicalResult {
  for (const auto& feature : getValue()) {
    const auto* iface =
        dyn_cast_if_present<FeatureDialectInterface>(feature.getNameDialect());
    if (!iface) {
      continue;
    }

    if (failed(iface->verify(op, feature))) {
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapsToAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyMapsToAttr(Operation* op,
                                       const NamedAttribute& attr)
    -> LogicalResult {
  if (isa<KTDFArchDialect>(op->getDialect())) {
    return emitIntrinsicError(op, attr) << "only valid on mappable ops";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OverlapsAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyOverlapsAttr(Operation* op,
                                         const NamedAttribute& attr)
    -> LogicalResult {
  const auto value = dyn_cast<OverlapsAttr>(attr.getValue());
  if (!value) {
    return emitIntrinsicError(op, attr) << "requires array attribute";
  }
  return success();
}

auto OverlapsAttr::contains(Attribute attr) const -> bool {
  return llvm::is_contained(getValue(), attr);
}

auto OverlapsAttr::overlaps(ArrayRef<Attribute> attrs) const -> bool {
  return llvm::any_of(attrs,
                      [&](Attribute attr) -> bool { return contains(attr); });
}

//===----------------------------------------------------------------------===//
// TransferGranularityAttr
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyTransferGranularityAttr(Operation* op,
                                                    const NamedAttribute& attr)
    -> LogicalResult {
  const auto emit_error = [&]() -> InFlightDiagnostic {
    return emitIntrinsicError(op, attr);
  };

  if (failed(verifyOn<Link>(emit_error, op))) {
    return failure();
  }

  const auto value = dyn_cast<TransferGranularityAttr>(attr.getValue());
  if (!value) {
    return emit_error() << "requires dense 64-bit integer array attribute";
  }
  return value.verify(emit_error);
}

auto TransferGranularityAttr::get(MLIRContext* context, ArrayRef<int64_t> value)
    -> TransferGranularityAttr {
  return cast<TransferGranularityAttr>(DenseI64ArrayAttr::get(context, value));
}

auto TransferGranularityAttr::verify(EmitErrorFn emit_error) const
    -> LogicalResult {
  llvm::SmallDenseSet<int64_t> values;
  for (auto value : asArrayRef()) {
    if (value < 0) {
      return emit_error() << "granularity must be >= 0";
    }
    if (!values.insert(value).second) {
      return emit_error() << "repeated granularity (" << value << ")";
    }
  }

  return success();
}

auto TransferGranularityAttr::contains(int64_t size) const -> bool {
  return llvm::is_contained(asArrayRef(), size);
}

//===----------------------------------------------------------------------===//
// AccessGranularityAttr
//===----------------------------------------------------------------------===//

auto AccessGranularityAttr::verify(EmitErrorFn emit_error) const
    -> LogicalResult {
  const auto size =
      dyn_cast_if_present<I64Attr>(DictionaryAttr::get(kSizeAttrName));
  if (!size) {
    return emit_error() << "attribute '" << kSizeAttrName
                        << "' requires 64-bit integer";
  }
  if (size.getValue() <= 0) {
    return emit_error() << "attribute '" << kSizeAttrName << "' must be >= 0";
  }

  const auto align = DictionaryAttr::get(kAlignAttrName);
  if (align) {
    const auto typed = dyn_cast<I64Attr>(align);
    if (!typed) {
      return emit_error() << "attribute '" << kAlignAttrName
                          << "' requires 64-bit integer";
    }
    if (typed.getValue() <= 0) {
      return emit_error() << "attribute '" << kAlignAttrName
                          << "' must be >= 0";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AccessGranularityListAttr
//===----------------------------------------------------------------------===//

auto AccessGranularityListAttr::fitAccess(size_t min_size_in_words,
                                          size_t max_align_in_words) const
    -> AccessGranularityAttr {
  AccessGranularityAttr result;
  auto best_size = AccessGranularityAttr::kMaxSize;
  auto best_align = max_align_in_words;

  for (const auto access : getValue()) {
    // Filter by size requirement.
    const auto size = access.getSizeInWords();
    if (size < min_size_in_words) {
      continue;
    }

    // Filter by alignment requirement.
    const auto align = access.getAlignInWords();
    if (align > max_align_in_words) {
      continue;
    }

    if (result) {
      if (size > best_size) {
        // Found transfer is smaller.
        continue;
      }
      if (size == best_size && align > best_align) {
        // Found transfer has same size but more permissive alignment.
        continue;
      }
    }

    result = access;
    best_size = size;
    best_align = align;
  }

  return result;
}

auto AccessGranularityListAttr::test(AccessGranularityListAttr required) const
    -> bool {
  return llvm::all_of(
      required, [&](AccessGranularityAttr required_access) -> bool {
        return fitAccess(required_access.getSizeInWords(),
                         required_access.getAlignInWords()) != nullptr;
      });
}

//===----------------------------------------------------------------------===//
// LoadStoreAttr
//===----------------------------------------------------------------------===//

auto LoadStoreAttr::verify(EmitErrorFn emit_error) const -> LogicalResult {
  const auto word_size = getAttr(kWordSizeAttrName);
  if (word_size) {
    const auto typed = dyn_cast<WordSizeMapAttr>(word_size);
    if (!typed) {
      return emit_error() << "attribute '" << kWordSizeAttrName
                          << "' requires map from attribute to 64-bit integer";
    }

    for (auto [space, size] : typed) {
      if (size.getValue() <= 0) {
        return emit_error() << "attribute '" << kWordSizeAttrName << "["
                            << space << "]' must be >= 0";
      }
    }
  }

  const auto access_granularity = getAttr(kAccessGranularityAttrName);
  if (access_granularity) {
    const auto typed = dyn_cast<AccessGranularityMapAttr>(access_granularity);
    if (!typed) {
      return emit_error() << "attribute '" << kAccessGranularityAttrName
                          << "' requires map from "
                             "attribute to array of dictionary attribtues";
    }

    for (const auto entry : typed) {
      const auto emit_space_error = [&]() {
        return emit_error() << "attribute '" << kAccessGranularityAttrName
                            << "[" << entry.first << "]' ";
      };
      for (const auto access : entry.second) {
        if (failed(access.verify(emit_space_error))) {
          return failure();
        }
      }
    }
  }

  return success();
}

auto LoadStoreAttr::test(LoadStoreAttr requirements) const -> bool {
  if (const auto required = requirements.getWordSize(); required) {
    for (const auto [space, required_word_size] : required) {
      const auto provided_word_size = getWordSize(space);
      if (provided_word_size !=
          static_cast<size_t>(required_word_size.getValue())) {
        return false;
      }
    }
  }

  if (const auto required = requirements.getAccessGranularity(); required) {
    for (const auto [space, required_accesses] : required) {
      if (required_accesses.empty()) {
        continue;
      }

      const auto provided_accesses = getAccessGranularity(space);
      if (!provided_accesses || !provided_accesses.test(required_accesses)) {
        return false;
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// feature::Compute
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureComputeAttr(Operation* op,
                                               const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::Compute, ExecutionUnitOp>(op, attr);
}

auto KTDFArchDialect::testFeatureCompute(Attribute provided,
                                         const Feature& required) -> bool {
  return provided == required.getValue();
}

//===----------------------------------------------------------------------===//
// feature::Load
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureLoadAttr(Operation* op,
                                            const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::Load, ExecutionUnitOp>(op, attr);
}

auto KTDFArchDialect::testFeatureLoad(Attribute provided,
                                      const Feature& required) -> bool {
  const auto required_value = cast<feature::Load>(required.getValue());
  const auto provided_value = cast<feature::Load>(provided);

  return provided_value.test(required_value);
}

//===----------------------------------------------------------------------===//
// feature::RegisterFile
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureRegisterFileAttr(Operation* op,
                                                    const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::RegisterFile, MemoryOp>(op, attr);
}

auto feature::RegisterFile::verify(EmitErrorFn emit_error) const
    -> LogicalResult {
  if (const auto maybe_word_size = getValue<I64Attr>(kWordSizeAttrName);
      maybe_word_size) {
    if (*maybe_word_size <= 0) {
      return emit_error() << "'" << kWordSizeAttrName << "' must be > 0";
    }
  } else if (getAttr(kWordSizeAttrName)) {
    return emit_error() << "attribute '" << kWordSizeAttrName
                        << "' requires 64-bit integer";
  }

  return success();
}

auto KTDFArchDialect::testFeatureRegisterFile(Attribute provided,
                                              const Feature& required) -> bool {
  const auto required_value = cast<feature::RegisterFile>(required.getValue());
  const auto provided_value = cast<feature::RegisterFile>(provided);

  return provided_value.test(required_value);
}

auto feature::RegisterFile::test(RegisterFile requirements) const -> bool {
  if (const auto required_word_size = requirements.getWordSize();
      required_word_size) {
    if (getWordSize() != required_word_size) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// feature::SIMD
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureSIMDAttr(Operation* op,
                                            const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::SIMD, ExecutionUnitOp>(op, attr);
}

auto feature::SIMD::verify(EmitErrorFn emit_error) const -> LogicalResult {
  const auto splat = getAttr(kSplatAttrName);
  if (splat && !isa<UnitAttr>(splat)) {
    return emit_error() << "attribute '" << kSplatAttrName << "' requires unit";
  }

  const auto zero_pad = getAttr(kZeroPadAttrName);
  if (zero_pad && !isa<UnitAttr>(zero_pad)) {
    return emit_error() << "attribute '" << kZeroPadAttrName
                        << "' requires unit";
  }

  const auto lanes = getAttr(kLanesAttrName);
  if (lanes && !isa<LanesAttr>(lanes)) {
    return emit_error() << "attribute '" << kLanesAttrName
                        << "' requires map from type to 64-bit integer";
  }

  return success();
}

auto KTDFArchDialect::testFeatureSIMD(Attribute provided,
                                      const Feature& required) -> bool {
  const auto required_value = cast<feature::SIMD>(required.getValue());
  const auto provided_value = cast<feature::SIMD>(provided);

  return provided_value.test(required_value);
}

auto feature::SIMD::test(feature::SIMD requirements) const -> bool {
  if (requirements.canSplat() && !canSplat()) {
    return false;
  }

  if (requirements.canZeroPad() && !canZeroPad()) {
    return false;
  }

  if (const auto required = requirements.getLanes(); required) {
    const auto required_lanes = required.getEntries();
    const auto provided_lanes = getLanes();

    if (!provided_lanes && !required_lanes.empty()) {
      return false;
    }

    if (llvm::any_of(required_lanes, [&](const auto& require) -> bool {
          return provided_lanes.getValue(require.first) <
                 require.second.getValue();
        })) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// feature::Store
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureStoreAttr(Operation* op,
                                             const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::Store, ExecutionUnitOp>(op, attr);
}

auto KTDFArchDialect::testFeatureStore(Attribute provided,
                                       const Feature& required) -> bool {
  const auto required_value = cast<feature::Store>(required.getValue());
  const auto provided_value = cast<feature::Store>(provided);

  return provided_value.test(required_value);
}

//===----------------------------------------------------------------------===//
// feature::Queue
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyFeatureQueueAttr(Operation* op,
                                             const NamedAttribute& attr)
    -> LogicalResult {
  return verifyFeature<feature::Queue, Link>(op, attr);
}

auto feature::Queue::verify(EmitErrorFn emit_error) const -> LogicalResult {
  const auto ordered = getAttr(kOrderedAttrName);
  if (ordered && !isa<UnitAttr>(ordered)) {
    return emit_error() << "attribute '" << kOrderedAttrName
                        << "' requires unit";
  }

  if (const auto maybe_size = getSize(); maybe_size) {
    if (*maybe_size <= 0) {
      return emit_error() << "'" << kSizeAttrName << "' must be > 0";
    }
  } else if (getAttr(kSizeAttrName)) {
    return emit_error() << "attribute '" << kSizeAttrName
                        << "' requires 64-bit integer";
  }

  if (const auto maybe_depth = getDepth(); maybe_depth) {
    if (*maybe_depth <= 0) {
      return emit_error() << "'" << kDepthAttrName << "' must be > 0";
    }
  } else if (getAttr(kDepthAttrName)) {
    return emit_error() << "attribute '" << kDepthAttrName
                        << "' requires 64-bit integer";
  }

  return success();
}

auto KTDFArchDialect::testFeatureQueue(Attribute provided,
                                       const Feature& required) -> bool {
  const auto required_value = cast<feature::Queue>(required.getValue());
  const auto provided_value = cast<feature::Queue>(provided);

  return provided_value.test(required_value);
}

auto feature::Queue::test(Queue requirements) const -> bool {
  if (requirements.isOrdered() && !isOrdered()) {
    return false;
  }

  if (const auto required = requirements.getSize(); required) {
    const auto provided = getSize();
    if (!provided || provided < required) {
      return false;
    }
  }

  if (const auto required = requirements.getDepth(); required) {
    const auto provided = getDepth();
    if (!provided || provided < required) {
      return false;
    }
  }

  return true;
}
