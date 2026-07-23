//===-- ResourceKinds.h -----------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEKINDS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEKINDS_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/AnalysisManager.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace mlir::ktdf_arch {

/// Analysis that collects the different kinds of resources in a device.
///
/// An invariant of `ktdf_arch` is that all resources annotated with the same
/// `kind` must be locally similar. This means that a single exemplar with a
/// given kind is representative of all other instances in the graph. With this
/// analysis, a minimal map from kinds to canonical exemplars is cached.
///
/// Note that local similarity is constraining downwards, but not upwards. In
/// other words, if an exemplar is found to have a certain child, all other
/// instances must have it too, but the same is not true for its parent. While
/// traversing the device, this analysis will also collect all known enclosing
/// parent kinds for the exemplars.
class ResourceKinds : public DeviceView {
 public:
  /// Caches information about a kind of resource.
  struct Kind {
    /*implicit*/ Kind() = default;
    explicit Kind(Resource exemplar) : exemplar_(exemplar) {}

    /// Gets the kind attribute.
    [[nodiscard]] auto getKind() const -> Attribute {
      return *this ? getExemplar().getKind() : nullptr;
    }
    /// @copydoc getKind()
    /*implicit*/ operator Attribute() const { return getKind(); }

    /// Gets the exemplar.
    [[nodiscard]] auto getExemplar() const -> Resource { return exemplar_; }
    /// @copydoc getExemplar()
    /*implicit*/ operator Resource() const { return getExemplar(); }

    /// Gets the known enclosing parent kinds.
    [[nodiscard]] auto getParentKinds() const
        -> const llvm::SmallPtrSet<Attribute, 4> {
      return parent_kinds_;
    }

    explicit operator bool() const { return exemplar_ != nullptr; }

    [[nodiscard]] auto operator<(const Kind& rhs) const -> bool {
      return getKind().getAsOpaquePointer() <
             rhs.getKind().getAsOpaquePointer();
    }

   private:
    friend class ResourceKinds;

    Resource exemplar_;
    llvm::SmallPtrSet<Attribute, 4> parent_kinds_;
  };

  /// Creates the ResourceKinds for the device declared by @p declaration .
  explicit ResourceKinds(DeviceOp declaration, AnalysisManager& analyses);

  /// Obtains an exemplar of @p kind , if any exists.
  ///
  /// @tparam ResourceType  Expected resource type.
  ///
  /// @retval ResourceType  Exemplar for @p kind .
  /// @retval nullptr       No resource of @p kind or of different type.
  template <class ResourceType = Resource>
  [[nodiscard]] auto getInstance(Attribute kind) const -> ResourceType {
    if (const auto& entry = map_.lookup(kind); entry) {
      return mlir::dyn_cast<ResourceType>(entry.getExemplar().getOperation());
    }
    return nullptr;
  }
  /// Obtains an exemplar of @p kind , if any exists.
  ///
  /// @retval Resource  Exemplar for @p kind .
  /// @retval nullptr   No resource of @p kind or of different type.
  [[nodiscard]] auto operator[](Attribute kind) const -> Resource {
    return getInstance(kind);
  }

  /// Collects all transitive parent kinds of @p kind .
  void getAncestors(Attribute kind,
                    llvm::SmallPtrSet<Attribute, 8>& result) const;
  /// @copydoc getAncestors(Attribute, llvm::SmallSet<Attribute, 8> &)
  [[nodiscard]] auto getAncestors(Attribute kind) const
      -> llvm::SmallPtrSet<Attribute, 8> {
    llvm::SmallPtrSet<Attribute, 8> result;
    getAncestors(kind, result);
    return result;
  }

  /// Collects all instances of @p kind .
  void getInstances(Attribute kind,
                    llvm::SmallVectorImpl<Resource>& result) const;
  /// @copydoc getInstances(Attribute, llvm::SmallVectorImpl<Resource> &)
  [[nodiscard]] auto getInstances(Attribute kind) const
      -> llvm::SmallVector<Resource> {
    llvm::SmallVector<Resource> result;
    getInstances(kind, result);
    return result;
  }

  /// Tries to get the @p PropertyAttr of an exemplar for @p kind .
  template <class PropertyAttr>
  [[nodiscard]] auto getProperty(Attribute kind) const -> PropertyAttr {
    if (auto resource = getInstance(kind); resource) {
      return resource.getProperty<PropertyAttr>();
    }

    return nullptr;
  }

  /// Tries to get the @p FeatureAttr of an exemplar for @p kind .
  template <class FeatureAttr>
  [[nodiscard]] auto getFeature(Attribute kind) const -> FeatureAttr {
    if (auto resource = getInstance(kind); resource) {
      return resource.getFeature<FeatureAttr>();
    }

    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using map_type = llvm::DenseMap<Attribute, Kind>;
  using value_type = Kind;
  using size_type = map_type::size_type;

  struct iterator
      : llvm::mapped_iterator_base<iterator, map_type::const_iterator,
                                   const value_type&> {
    using llvm::mapped_iterator_base<iterator, map_type::const_iterator,
                                     const value_type&>::mapped_iterator_base;

    [[nodiscard]] static auto mapElement(const map_type::value_type& pair)
        -> const value_type& {
      return pair.second;
    }
  };

  [[nodiscard]] auto empty() const -> bool { return map_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return map_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return map_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return map_.end(); }

  [[nodiscard]] auto find(Attribute kind) const -> iterator {
    return map_.find(kind);
  }

 private:
  map_type map_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_RESOURCEKINDS_H_
