//===-- Mapping.h -----------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_MAPPING_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_MAPPING_H_

#include <llvm/ADT/PointerUnion.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/AnalysisManager.h>

#include <cstddef>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceIds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

//===----------------------------------------------------------------------===//
// ResourceSpec
//===----------------------------------------------------------------------===//

namespace mlir::ktdf_arch {

/// Specifier that selects an architecture graph resource.
class ResourceSpec : public llvm::PointerUnion<Attribute, Operation*> {
 public:
  using base = llvm::PointerUnion<Attribute, Operation*>;

  /// Initializes a specifier that matches any exemplar of @p kind .
  [[nodiscard]] static auto kind(Attribute kind) -> ResourceSpec {
    assert(!isa<SymbolRefAttr>(kind));
    return ResourceSpec(kind);
  }

  /// Initializes a specifier that matches nothing.
  /*implicit*/ ResourceSpec(std::nullptr_t = nullptr) : base(nullptr) {}
  /// Initializes a specifier that matches @p resource exactly.
  /*implicit*/ ResourceSpec(Resource resource) : base(resource) {}

  /// Gets the kind of resource specified, if any.
  [[nodiscard]] auto getKind() const -> Attribute;

  void print(raw_ostream& os) const;
  friend auto operator<<(raw_ostream& os, ResourceSpec spec) -> raw_ostream& {
    spec.print(os);
    return os;
  }

 private:
  explicit ResourceSpec(Attribute kind) : base(kind) {}
};

}  // namespace mlir::ktdf_arch

template <typename To>
struct llvm::CastInfo<To, mlir::ktdf_arch::ResourceSpec>
    : CastInfo<To, mlir::ktdf_arch::ResourceSpec::base> {};

// Allow casting to mlir::ktdf_arch::Resource directly.
template <>
struct llvm::CastInfo<mlir::ktdf_arch::Resource, mlir::ktdf_arch::ResourceSpec>
    : DefaultDoCastIfPossible<
          mlir::ktdf_arch::Resource, mlir::ktdf_arch::ResourceSpec,
          CastInfo<mlir::ktdf_arch::Resource, mlir::ktdf_arch::ResourceSpec>> {
  using From = mlir::ktdf_arch::ResourceSpec;

  static auto isPossible(From& from) -> bool {
    return isa<mlir::Operation*>(from);
  }

  static auto doCast(From& from) -> mlir::ktdf_arch::Resource {
    assert(isPossible(from) && "cast to an incompatible type!");
    return cast<mlir::ktdf_arch::Resource>(cast<mlir::Operation*>(from));
  }

  static auto castFailed() -> mlir::ktdf_arch::Resource { return nullptr; }
};

template <typename To>
struct llvm::CastInfo<To, const mlir::ktdf_arch::ResourceSpec>
    : ConstStrippingForwardingCast<
          To, const mlir::ktdf_arch::ResourceSpec,
          CastInfo<To, mlir::ktdf_arch::ResourceSpec>> {};

// Allow using ResourceSpec as DenseMap keys.
template <>
struct llvm::DenseMapInfo<mlir::ktdf_arch::ResourceSpec> {
  [[nodiscard]] static auto getEmptyKey() -> mlir::ktdf_arch::ResourceSpec {
    return mlir::ktdf_arch::ResourceSpec::kind(
        DenseMapInfo<mlir::Attribute>::getEmptyKey());
  }

  [[nodiscard]] static auto getTombstoneKey() -> mlir::ktdf_arch::ResourceSpec {
    return mlir::ktdf_arch::ResourceSpec::kind(
        DenseMapInfo<mlir::Attribute>::getTombstoneKey());
  }

  [[nodiscard]] static auto getHashValue(
      const mlir::ktdf_arch::ResourceSpec& preds) {
    return DenseMapInfo<mlir::ktdf_arch::ResourceSpec::base>::getHashValue(
        preds);
  }

  [[nodiscard]] static bool isEqual(const mlir::ktdf_arch::ResourceSpec& lhs,
                                    const mlir::ktdf_arch::ResourceSpec& rhs) {
    return lhs == rhs;
  }
};

// Must be out of line because it requires the casting machinery.
inline auto mlir::ktdf_arch::ResourceSpec::getKind() const -> Attribute {
  if (auto resource =
          llvm::dyn_cast_if_present<mlir::ktdf_arch::Resource>(*this);
      resource) {
    return resource.getKind();
  }
  return llvm::dyn_cast_if_present<mlir::Attribute>(*this);
}

//===----------------------------------------------------------------------===//
// Mapping
//===----------------------------------------------------------------------===//

namespace mlir::ktdf_arch {

/// Adaptor that allows accessing the mapping between operations and resources.
///
/// Mappings between operations and resources are encoded in the IR, as values
/// of the `ktdf_arch.maps_to` intrinsic property attribute. This is so that
/// they are preserved between passes and can be set and queried by different
/// means, such as declarative rewrites.
///
/// By convention, a mapping is always implicitly relative to some device,
/// usually the only device in the module. By constructing a Mapping adapter,
/// this device information is stored so that the attribute can be resolved to
/// actual Resources, and such that Resources can be pinned to unique names.
class Mapping {
 public:
  /// Determines whether @p op is mappable.
  ///
  /// A mappable operation is not part of the `ktdf_arch` dialect, and is not
  /// nested under a `ktdf_arch.device` operation.
  [[nodiscard]] static auto isMappable(Operation* op) -> bool;

  /// Gets the closest mapping on @p mappable , if any.
  ///
  /// Walks the IR upwards from @p mappable , returning the value of the first
  /// attribute named `ktdf_arch.maps_to` found.
  ///
  /// @return Pair of the mapped ancestor and its mapping, or `nullptr`.
  [[nodiscard]] static auto get(Operation* op)
      -> std::pair<Operation*, MapsToAttr>;
  /// Sets the mapping on @p mappable to @p maps_to .
  ///
  /// @pre  `isMappable(mappable)`
  static void set(Operation* mappable, MapsToAttr maps_to) {
    assert(isMappable(mappable));
    setProperty(mappable, maps_to);
  }
  static void unset(Operation* mappable) {
    set(mappable, MapsToAttr::unmapped(mappable->getContext()));
  }

  /// Creates a Mapping relative to @p device .
  explicit Mapping(DeviceOp device, AnalysisManager analyses);

  /// Looks up the Resource @p maps_to refers to, if any.
  [[nodiscard]] auto lookup(MapsToAttr maps_to) const -> Resource;
  /// @copydoc lookup(MapsToAttr)
  [[nodiscard]] auto operator[](MapsToAttr maps_to) const -> Resource {
    return lookup(maps_to);
  }
  /// Looks up the Resource @p mappable is mapped to, if any.
  [[nodiscard]] auto lookup(Operation* mappable) const -> Resource {
    if (auto [op, attr] = get(mappable); attr) {
      return lookup(attr);
    }
    return nullptr;
  }
  /// @copydoc lookup(Operation*)
  [[nodiscard]] auto operator[](Operation* mappable) const -> ResourceSpec {
    return lookup(mappable);
  }

  /// Maps @p mappable to the Resource given by @p maps_to .
  void map(Operation* mappable, ResourceSpec maps_to);

  /// Gets the underlying Device the Mapping is relative to.
  [[nodiscard]] auto getDevice() const -> const Device& { return device_; }
  /// Gets the read-only ResourceIds of the device.
  [[nodiscard]] auto byId() const -> const ResourceIds& { return by_id_; }
  /// Gets the read-only ResourceKinds of the device.
  [[nodiscard]] auto byKind() const -> const ResourceKinds& { return by_kind_; }

 private:
  DeviceRef device_;
  ResourceIds& by_id_;
  ResourceKinds& by_kind_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_MAPPING_H_
