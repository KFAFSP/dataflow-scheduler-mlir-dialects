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
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/AnalysisManager.h>

#include <cstddef>
#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceIds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

//===----------------------------------------------------------------------===//
// ResourceSpec
//===----------------------------------------------------------------------===//

namespace mlir::ktdf_arch {

/// Specifier that selects an architecture graph resource.
class ResourceSpec : public llvm::PointerUnion<Resource, KindAttr> {
 public:
  using base = llvm::PointerUnion<Resource, KindAttr>;

  /// Initializes a specifier that matches nothing.
  /*implicit*/ ResourceSpec(std::nullptr_t = nullptr) : base(nullptr) {}
  /// Initializes a specifier that matches @p kind .
  /*implicit*/ ResourceSpec(KindAttr kind) : base(kind) {}
  /// Initializes a specifier that matches @p resource exactly.
  /*implicit*/ ResourceSpec(Resource resource) : base(resource) {}
  /// @copydoc ResourceSpec(Resource)
  /*implicit*/ ResourceSpec(Link resource) : base(resource) {}
  /// @copydoc ResourceSpec(Resource)
  /*implicit*/ ResourceSpec(Node resource) : base(resource) {}
  /// @copydoc ResourceSpec(Resource)
  template <class Op, class = std::enable_if_t<
                          std::is_base_of_v<Resource::Trait<Op>, Op>, void>>
  /*implicit*/ ResourceSpec(Op resource) : base(resource) {}

  /// Gets the kind of resource specified, if any.
  [[nodiscard]] auto getKind() const -> KindAttr;

  void print(raw_ostream& os) const;
  friend auto operator<<(raw_ostream& os, ResourceSpec spec) -> raw_ostream& {
    spec.print(os);
    return os;
  }
};

}  // namespace mlir::ktdf_arch

template <typename To>
struct llvm::CastInfo<To, mlir::ktdf_arch::ResourceSpec>
    : CastInfo<To, mlir::ktdf_arch::ResourceSpec::base> {};
template <typename To>
struct llvm::CastInfo<To, const mlir::ktdf_arch::ResourceSpec>
    : ConstStrippingForwardingCast<
          To, const mlir::ktdf_arch::ResourceSpec,
          CastInfo<To, mlir::ktdf_arch::ResourceSpec>> {};

// Allow using ResourceSpec as DenseMap keys.
template <>
struct llvm::DenseMapInfo<mlir::ktdf_arch::ResourceSpec> {
  [[nodiscard]] static auto getEmptyKey() -> mlir::ktdf_arch::ResourceSpec {
    return DenseMapInfo<mlir::ktdf_arch::KindAttr>::getEmptyKey();
  }

  [[nodiscard]] static auto getTombstoneKey() -> mlir::ktdf_arch::ResourceSpec {
    return DenseMapInfo<mlir::ktdf_arch::KindAttr>::getTombstoneKey();
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
inline auto mlir::ktdf_arch::ResourceSpec::getKind() const -> KindAttr {
  if (auto resource = llvm::dyn_cast_if_present<Resource>(*this); resource) {
    return resource.getKind();
  }
  return llvm::dyn_cast_if_present<KindAttr>(*this);
}

//===----------------------------------------------------------------------===//
// Mapping
//===----------------------------------------------------------------------===//

namespace mlir::ktdf_arch {

/// Adaptor class that manages the mapping from Mappable ops to Resources.
///
/// This mapping is encoded in the IR, usually via the `ktdf_arch.maps_to`
/// discardable dialect attribute. However, this attribute representation must
/// be resolved to Resources, which are relative to a Device. As a result, this
/// class does not store any of the mapping state, and only holds a DeviceRef.
///
/// Derived classes can override the behavior of map() and resolve() to
/// customize queries and updates to the mapping of the IR.
class Mapping {
 public:
  using Resources = llvm::SmallPtrSet<
      Resource,
      llvm::CalculateSmallVectorDefaultInlinedElements<Resource>::value>;

  /// Creates a Mapping relative to @p device .
  explicit Mapping(const DeviceRef& device);

  virtual ~Mapping() = default;

  /// Looks up the Resource @p maps_to refers to, if any.
  ///
  /// @tparam ResourceType  Expected resource type.
  ///
  /// @retval ResourceType  Instance for @p maps_to .
  /// @retval nullptr       No resource for @p maps_to or of different type.
  template <class ResourceType = Resource>
  [[nodiscard]] auto lookup(ResourceSpec maps_to) const -> ResourceType {
    if (const auto kind = dyn_cast<KindAttr>(maps_to); kind) {
      return by_kind_.getInstance<ResourceType>(kind);
    }
    auto resource = dyn_cast_if_present<Resource>(maps_to);
    if constexpr (std::is_same_v<ResourceType, Resource>) {
      return resource;
    } else {
      return resource ? dyn_cast<ResourceType>(resource.getOperation())
                      : nullptr;
    }
  }
  /// Looks up the Resource @p maps_to refers to, if any.
  [[nodiscard]] auto operator[](ResourceSpec maps_to) const -> Resource {
    return lookup(maps_to);
  }
  /// @copydoc lookup(ResourceSpec)
  template <class ResourceType = Resource>
  [[nodiscard]] auto lookup(ResourceSpecAttr maps_to) const -> ResourceType {
    if (auto id_ref = dyn_cast<FlatSymbolRefAttr>(maps_to); id_ref) {
      return by_id_.lookup<ResourceType>(id_ref.getAttr());
    }
    return by_kind_.getInstance<ResourceType>(cast<KindAttr>(maps_to));
  }
  /// Looks up the Resource @p maps_to refers to, if any.
  [[nodiscard]] auto operator[](ResourceSpecAttr maps_to) const -> Resource {
    return lookup(maps_to);
  }

  /// Resolves the resources reffered to by @p maps_to .
  ///
  /// @return Fails if any resource specifier could not be resolved.
  auto resolve(MapsToAttr maps_to, SmallPtrSetImpl<Resource>& resources) const
      -> LogicalResult;
  /// Resolves the resources @p mappable is mapped to, if any.
  ///
  /// @return Fails if any resource specifier could not be resolved.
  virtual auto resolve(Mappable mappable,
                       SmallPtrSetImpl<Resource>& resources) const
      -> LogicalResult {
    if (const auto mapping = mappable.getOrInheritMapsTo().second; mapping) {
      return resolve(mapping, resources);
    }
    return success();
  }
  /// Resolves the resources @p op is mapped to, if any.
  ///
  /// @return Fails if @p op isn't mappable or any resource spec is unresolved.
  auto resolve(Operation* op, SmallPtrSetImpl<Resource>& resources) const
      -> LogicalResult {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return resolve(mappable, resources);
    }
    return failure();
  }
  /// Resolves the singular Resource @p mappable is mapped to, if any.
  ///
  /// @tparam ResourceType  Expected resource type.
  ///
  /// @retval ResourceType  Singular resource @p mappable is mapped to.
  /// @retval nullptr       Invalid or undefined mapping.
  template <class ResourceType = Resource>
  [[nodiscard]] auto resolve(Mappable mappable) -> ResourceType {
    SmallPtrSet<Resource, 1> resources;
    if (failed(resolve(mappable, resources)) || resources.size() != 1) {
      return nullptr;
    }
    auto resource = *resources.begin();
    if constexpr (std::is_same_v<ResourceType, Resource>) {
      return resource;
    } else {
      return dyn_cast<ResourceType>(resource.getOperation());
    }
  }
  /// Resolves the singular Resource @p op is mapped to, if any.
  ///
  /// @tparam ResourceType  Expected resource type.
  ///
  /// @retval ResourceType  Singular resource @p mappable is mapped to.
  /// @retval nullptr       Invalid or undefined mapping.
  template <class ResourceType = Resource>
  [[nodiscard]] auto resolve(Operation* op) -> ResourceType {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return resolve<ResourceType>(mappable);
    }
    return nullptr;
  }

  /// Creates a ResourceSpecAttr from @p maps_to .
  ///
  /// @retval ResourceSpecAttr  Attribute that references @p maps_to .
  /// @retval nullptr           @p maps_to is not part of the device.
  virtual auto map(ResourceSpec maps_to) -> ResourceSpecAttr;
  /// Creates a MapsToAttr from @p maps_to .
  ///
  /// @retval MapsToAttr  Attribute that references @p maps_to .
  /// @retval nullptr     Some @p maps_to resource is not part of the device.
  auto map(ArrayRef<ResourceSpec> maps_to) -> MapsToAttr;
  /// Maps @p mappable to the Resources given by @p maps_to .
  ///
  /// The caller is responsible for ensuring that @p maps_to is a valid mapping
  /// for @p mappable . The operation may customize the behavior of setting the
  /// mapping, which is allowed to fail.
  ///
  /// @return Whether @p mappable has updated its mapping.
  virtual auto map(Mappable mappable, ArrayRef<ResourceSpec> maps_to)
      -> LogicalResult;
  /// Maps @p op to the Resources given by @p maps_to .
  ///
  /// See map(Mappable, ArrayRef<ResourceSpec>) for more information.
  ///
  /// @return Whether @p op has updated its mapping.
  auto map(Operation* op, ArrayRef<ResourceSpec> maps_to) -> LogicalResult {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return map(mappable, maps_to);
    }
    return failure();
  }

  /// Resolves the Resources @p mappable is mapped to, mapping it to @p fallback
  /// if it has no established mapping.
  ///
  /// See map(Mappable, ArrayRef<ResourceSpec>) for more details on mapping.
  ///
  /// @retval Resources Resources that @p mappable is mapped to.
  /// @retval failure   Invalid mapping or @p fallback .
  auto getOrMap(Mappable mappable, ArrayRef<ResourceSpec> fallback)
      -> FailureOr<Resources>;
  /// Resolves the Resources @p op is mapped to, mapping it to @p fallback if it
  /// has no established mapping.
  ///
  /// See map(Mappable, ArrayRef<ResourceSpec>) for more details on mapping.
  ///
  /// @retval Resources Resources that @p op is mapped to.
  /// @retval failure   Invalid mapping or @p fallback .
  auto getOrMap(Operation* op, ArrayRef<ResourceSpec> fallback)
      -> FailureOr<Resources> {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return getOrMap(mappable, fallback);
    }
    return failure();
  }
  /// Resolves the singular Resource @p mappable is mapped to, mapping it to
  /// @p fallback if it has no established mapping.
  ///
  /// See map(Mappable, ArrayRef<ResourceSpec>) for more details on mapping.
  ///
  /// @retval ResourceType  Resource that @p mappable is mapped to.
  /// @retval nullptr       Invalid mapping or @p fallback .
  template <class ResourceType = Resource>
  auto getOrMap(Mappable mappable, ResourceSpec fallback) -> ResourceType {
    SmallPtrSet<Resource, 1> result;
    if (failed(resolve(mappable, result)) || result.size() > 1) {
      return nullptr;
    }

    Resource resource;
    if (result.empty()) {
      resource = lookup(fallback);
      if constexpr (!std::is_same_v<ResourceType, Resource>) {
        if (!isa<ResourceType>(resource.getOperation())) {
          return nullptr;
        }
      }

      if (failed(map(mappable, {fallback}))) {
        return nullptr;
      }
    } else {
      resource = *result.begin();
    }

    if constexpr (std::is_same_v<ResourceType, Resource>) {
      return resource;
    } else {
      return resource ? dyn_cast<ResourceType>(resource.getOperation())
                      : nullptr;
    }
  }
  /// Resolves the singular Resource @p op is mapped to, mapping it to
  /// @p fallback if it has no established mapping.
  ///
  /// See map(Mappable, ArrayRef<ResourceSpec>) for more details on mapping.
  ///
  /// @retval ResourceType  Resource that @p op is mapped to.
  /// @retval nullptr       Invalid mapping or @p fallback .
  template <class ResourceType = Resource>
  auto getOrMap(Operation* op, ResourceSpec fallback) -> ResourceType {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return getOrMap<ResourceType>(mappable, fallback);
    }
    return nullptr;
  }

  /// Verifies the mapping of @p mappable .
  auto verify(Mappable mappable) const -> LogicalResult;
  /// Verifies the mapping of @p op .
  auto verify(Operation* op) const -> LogicalResult {
    if (auto mappable = dyn_cast<Mappable>(op); mappable) {
      return verify(mappable);
    }
    return success();
  }

  /// Gets the MLIR context.
  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return device_->getContext();
  }
  /// Gets the underlying Device the Mapping is relative to.
  [[nodiscard]] auto getDevice() const -> const Device& { return device_; }
  /// Gets the read-only ResourceIds of the device.
  [[nodiscard]] auto byId() const -> const ResourceIds& { return by_id_; }
  /// Gets the read-only ResourceKinds of the device.
  [[nodiscard]] auto byKind() const -> const ResourceKinds& { return by_kind_; }

 private:
  /// Verifies the mapping of @p mappable to @p resources .
  virtual auto verifyImpl(Mappable mappable, ArrayRef<Resource> resources) const
      -> LogicalResult;

  const DeviceRef& device_;
  ResourceIds& by_id_;
  const ResourceKinds& by_kind_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_MAPPING_H_
