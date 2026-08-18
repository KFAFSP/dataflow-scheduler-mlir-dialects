//===-- KTDFArchIntrinsics.h ------------------------------------*- c++ -*-===//
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
// This file defines the intrisics of the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>

#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"

namespace mlir::ktdf_arch {

using GetAttrNameFn = StringAttr (KTDFArchDialect::*)() const;
using EmitErrorFn = function_ref<InFlightDiagnostic()>;

/// Non-templated base class for feature attributes.
struct FeatureAttrBase : Attribute {
  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    return isa<UnitAttr, DictionaryAttr>(attr);
  }
  [[nodiscard]] static auto classof(UnitAttr /*attr*/) -> bool { return true; }
  [[nodiscard]] static auto classof(DictionaryAttr /*attr*/) -> bool {
    return true;
  }

  using Attribute::Attribute;

  /*implicit*/ FeatureAttrBase(UnitAttr attr) : Attribute(attr.getImpl()) {}
  /*implicit*/ FeatureAttrBase(DictionaryAttr attr) : Attribute(attr) {}

  /// Verifies this feature attribute.
  [[nodiscard]] auto verify(EmitErrorFn /*emit_error*/) const -> LogicalResult {
    return success();
  }

  /// Determines whether this feature satisfies @p requirements .
  [[nodiscard]] auto test(Attribute requirements) const -> bool {
    return *this == requirements || isa<UnitAttr>(requirements);
  }

  /// Determines whether this feature has no nested attributes.
  [[nodiscard]] auto empty() const -> bool {
    return !*this || isa<UnitAttr>(*this) ||
           cast<DictionaryAttr>(*this).empty();
  }

  /// Obtains the nested @p Attr with @p name , if any.
  template <class Attr = Attribute>
  [[nodiscard]] auto getAttr(StringRef name) const -> Attr {
    const auto dict = dyn_cast_if_present<DictionaryAttr>(*this);
    return dict ? dict.template getAs<Attr>(name) : nullptr;
  }

  /// Obtains the value of the nested @p Attr with @p name , if any.
  template <class Attr = Attribute>
  [[nodiscard]] auto getValue(StringRef name) const
      -> std::optional<typename Attr::ValueType> {
    if (const auto attr = getAttr<Attr>(name); attr) {
      return attr.getValue();
    }

    return std::nullopt;
  }
};

/// Base class for intrinsic attributes.
///
/// @tparam GetAttrName     Pointer to the cached name getter in the dialect.
/// @tparam AttrConstraint  Underlying attribute type constraint.
template <GetAttrNameFn GetAttrName, class AttrConstraint = Attribute>
struct IntrinsicAttr : AttrConstraint {
  static_assert(std::is_base_of_v<Attribute, AttrConstraint>);

  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    return isa<AttrConstraint>(attr);
  }

  using AttrConstraint::AttrConstraint;

  /// Gets the cached attribute name.
  [[nodiscard]] static auto getAttrName(MLIRContext* context) -> StringAttr {
    const auto* dialect = context->getLoadedDialect<KTDFArchDialect>();
    return (dialect->*GetAttrName)();
  }
  /// Gets the cached attribute name.
  [[nodiscard]] auto getAttrName() const -> StringAttr {
    return getAttrName(static_cast<const Attribute&>(*this).getContext());
  }
};

/// Base class for intrinsic features.
///
/// @tparam GetAttrName     Pointer to the cached name getter in the dialect.
/// @tparam AttrConstaint   Underlying attribute type constraint.
template <GetAttrNameFn GetAttrName, class AttrConstraint = FeatureAttrBase>
struct FeatureAttr : IntrinsicAttr<GetAttrName, AttrConstraint> {
  using IntrinsicAttr<GetAttrName, AttrConstraint>::IntrinsicAttr;
};

//===----------------------------------------------------------------------===//
// Intrinsic Properties
//===----------------------------------------------------------------------===//

/// Indicates the bandwidth of a link in bytes per unit time.
struct BandwidthAttr
    : IntrinsicAttr<&KTDFArchDialect::getBandwidthAttrName, I64Attr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Gets the canonical BandwidthAttr for @p value .
  static auto get(MLIRContext* context, int64_t value) -> BandwidthAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;
};

/// Indicates the set of provided/required features of an operation.
struct FeaturesAttr
    : IntrinsicAttr<&KTDFArchDialect::getFeaturesAttrName, DictionaryAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  auto verify(Operation* op) const -> LogicalResult;
};

/// Indicates the architecture element an operation is mapped to.
struct MapsToAttr
    : IntrinsicAttr<&KTDFArchDialect::getMapsToAttrName, Attribute> {
  using IntrinsicAttr::IntrinsicAttr;
};

/// Indicates a set of mutexes that allocation of this resource must obey.
struct OverlapsAttr
    : IntrinsicAttr<&KTDFArchDialect::getOverlapsAttrName, ArrayAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Determines whether @p attr is in the set.
  [[nodiscard]] auto contains(Attribute attr) const -> bool;

  /// Determines whether @p attrs overlap any in the set.
  [[nodiscard]] auto overlaps(ArrayRef<Attribute> attrs) const -> bool;
  /// Determines whether @p attr overlaps with the set.
  [[nodiscard]] auto overlaps(OverlapsAttr attr) const -> bool {
    return overlaps(attr.getValue());
  }
};

/// Indicates the permissible sizes of individual transfers on a link.
struct TransferGranularityAttr
    : IntrinsicAttr<&KTDFArchDialect::getTransferGranularityAttrName,
                    DenseI64ArrayAttr> {
  using IntrinsicAttr::IntrinsicAttr;

  /// Gets the canonical TransferGranularityAttr for @p value .
  static auto get(MLIRContext* context, ArrayRef<int64_t> value)
      -> TransferGranularityAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  /// Determines whether @p size is a permissible transfer size.
  [[nodiscard]] auto contains(int64_t size) const -> bool;
};

//===----------------------------------------------------------------------===//
// Feature Properties
//===----------------------------------------------------------------------===//

/// Describes the granularity of a memory access.
struct AccessGranularityAttr : DictionaryAttr {
  static constexpr size_t kMaxSize = SIZE_MAX;
  static constexpr size_t kMinAlign = 1;
  static constexpr size_t kMaxAlign = SIZE_MAX;
  static constexpr StringLiteral kSizeAttrName = "size_in_words";
  static constexpr StringLiteral kAlignAttrName = "align_in_words";

  using DictionaryAttr::DictionaryAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  /// Gets the access size in data words.
  [[nodiscard]] auto getSizeInWords() const -> size_t {
    return DictionaryAttr::getAs<I64Attr>(kSizeAttrName).getValue();
  }
  /// Gets the access alignment in data words.
  [[nodiscard]] auto getAlignInWords() const -> size_t {
    if (const auto attr = DictionaryAttr::getAs<I64Attr>(kAlignAttrName);
        attr) {
      return attr.getValue();
    }
    return kMinAlign;
  }
};

/// Stores a list of memory accesses.
struct AccessGranularityListAttr : TypedArrayAttr<AccessGranularityAttr> {
  using TypedArrayAttr::TypedArrayAttr;

  /// Finds the smallest access that fits @p min_size_in_words with
  /// @p max_align_in_words .
  ///
  /// If there are multiple candidates of exact @p min_size , then the
  /// candidate with the smallest (most permissive) alignment is chosen.
  ///
  /// @param  min_size_in_words     Minimum transfer size in data words.
  /// @param  max_align_in_words    Maximum address alignment in data words
  ///                               (defaults to the most permissive alignment).
  ///
  /// @retval nullptr               No fitting access found.
  /// @retval AccessGranularityAttr Best access found.
  [[nodiscard]] auto fitAccess(
      size_t min_size_in_words,
      size_t max_align_in_words = AccessGranularityAttr::kMaxAlign) const
      -> AccessGranularityAttr;

  [[nodiscard]] auto test(AccessGranularityListAttr required) const -> bool;
};

/// Maps memory spaces to lists of memory accesses.
using AccessGranularityMapAttr =
    TypedMapAttr<Attribute, AccessGranularityListAttr>;

/// Holds information shared between the load & store feature.
struct LoadStoreAttr : FeatureAttrBase {
  static constexpr StringLiteral kAccessGranularityAttrName =
      "access_granularity";
  static constexpr StringLiteral kWordSizeAttrName = "word_size";

  using WordSizeMapAttr = TypedMapAttr<Attribute, I64Attr>;

  using FeatureAttrBase::FeatureAttrBase;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  [[nodiscard]] auto test(LoadStoreAttr requirements) const -> bool;

  /// Gets a map from memory spaces to access granularities.
  [[nodiscard]] auto getAccessGranularity() const -> AccessGranularityMapAttr {
    return getAttr<AccessGranularityMapAttr>(kAccessGranularityAttrName);
  }
  /// Gets the access granularities for @p memory_space .
  [[nodiscard]] auto getAccessGranularity(Attribute memory_space) const
      -> AccessGranularityListAttr {
    if (const auto access_granularity = getAccessGranularity();
        access_granularity) {
      return access_granularity.getAttr(memory_space);
    }
    return nullptr;
  }

  /// Gets a map from memory spaces to data word sizes.
  [[nodiscard]] auto getWordSize() const -> WordSizeMapAttr {
    return getAttr<WordSizeMapAttr>(kWordSizeAttrName);
  }
  /// Gets the data word size for @p memory_space .
  [[nodiscard]] auto getWordSize(Attribute memory_space) const -> size_t {
    if (const auto word_size = getWordSize(); word_size) {
      return static_cast<size_t>(word_size.getValue(memory_space).value_or(1));
    }
    return 1;
  }
};

//===----------------------------------------------------------------------===//
// Intrinsic Features
//===----------------------------------------------------------------------===//

namespace feature {

/// Indicates that the execution unit can implement arbitrary programs.
struct Compute : FeatureAttr<&KTDFArchDialect::getFeatureComputeAttrName> {
  using FeatureAttr::FeatureAttr;
};

// Indicates that the execution unit can perform load operations.
struct Load
    : FeatureAttr<&KTDFArchDialect::getFeatureLoadAttrName, LoadStoreAttr> {
  using FeatureAttr::FeatureAttr;
};

/// Indicates that the execution unit has SIMD lanes.
struct SIMD : FeatureAttr<&KTDFArchDialect::getFeatureSIMDAttrName> {
  static constexpr StringLiteral kSplatAttrName = "splat";
  static constexpr StringLiteral kZeroPadAttrName = "zero_pad";
  static constexpr StringLiteral kLanesAttrName = "lanes";

  using LanesAttr = TypedMapAttr<TypeAttr, I64Attr>;

  using FeatureAttr::FeatureAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  [[nodiscard]] auto test(SIMD requirements) const -> bool;

  /// Determines whether the unit can form splat vectors.
  [[nodiscard]] auto canSplat() const -> bool {
    return getAttr<UnitAttr>(kSplatAttrName) != nullptr;
  }

  /// Determines whether the unit can pad vectors with zeros.
  [[nodiscard]] auto canZeroPad() const -> bool {
    return getAttr<UnitAttr>(kZeroPadAttrName) != nullptr;
  }

  /// Gets the number of lanes per scalar type.
  [[nodiscard]] auto getLanes() const -> LanesAttr {
    return getAttr<LanesAttr>(kLanesAttrName);
  }
  /// Gets the number of lanes for @p scalar_type .
  ///
  /// @returns  Number of lanes for @p scalar_type , or 0 if not supported.
  [[nodiscard]] auto getLanes(Type scalar_type) const -> int64_t {
    if (const auto lanes = getLanes(); lanes) {
      return lanes.getValue(TypeAttr::get(scalar_type)).value_or(0);
    }
    return 0;
  }
};

// Indicates that the execution unit can perform store operations.
struct Store
    : FeatureAttr<&KTDFArchDialect::getFeatureStoreAttrName, LoadStoreAttr> {
  using FeatureAttr::FeatureAttr;
};

/// Indicates that the link can have multiple transactions in flight.
struct Queue : FeatureAttr<&KTDFArchDialect::getFeatureQueueAttrName> {
  static constexpr StringLiteral kOrderedAttrName = "ordered";
  static constexpr StringLiteral kSizeAttrName = "size";
  static constexpr StringLiteral kDepthAttrName = "depth";

  using FeatureAttr::FeatureAttr;

  auto verify(EmitErrorFn emit_error) const -> LogicalResult;

  [[nodiscard]] auto test(Queue requirements) const -> bool;

  /// Gets a value indicating whether the transactions must complete in order.
  [[nodiscard]] auto isOrdered() const -> bool {
    return getAttr<UnitAttr>(kOrderedAttrName) != nullptr;
  }

  /// Gets a value indicating the total size of in flight transactions.
  [[nodiscard]] auto getSize() const -> std::optional<int64_t> {
    return getValue<I64Attr>(kSizeAttrName);
  }

  /// Gets a value indicating how many transactions can be in flight.
  [[nodiscard]] auto getDepth() const -> std::optional<int64_t> {
    return getValue<I64Attr>(kDepthAttrName);
  }
};

}  // namespace feature

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_H_
