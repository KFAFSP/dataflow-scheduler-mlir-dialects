//===-- Query.h -------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_QUERY_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_QUERY_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Attributes.h>

#include <algorithm>
#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"

namespace mlir::ktdf_arch {

/// Helper for performing a query over resources.
///
/// A query is a non-owning reference to a range of results that can be mutated.
/// Whenever the query is refined, using one of its fluent methods, the range is
/// changed such that items not matching the new filter are moved towards the
/// end. This means the order of results does not remain stable for previous
/// queries, but the order of elements in the final result will be stable with
/// respect to the original input.
template <class ResourceType = Resource>
class Query : public MutableArrayRef<ResourceType> {
  using base = MutableArrayRef<ResourceType>;

 public:
  /// Start a query over @p results .
  /*implicit*/ Query(MutableArrayRef<ResourceType> results = {})
      : base(results) {}

  /// Removes resources not matching @p predicate .
  [[nodiscard]] auto where(function_ref<bool(ResourceType)> predicate)
      -> Query {
    auto* const begin = base::begin();
    auto* const end = std::partition(begin, base::end(), predicate);
    return Query(base(begin, std::distance(begin, end)));
  }

  /// Removes duplicate resources.
  [[nodiscard]] auto unique() -> Query {
    auto* const begin = base::begin();
    llvm::DenseSet<Resource> seen;
    auto* const end =
        std::remove_if(begin, base::end(), [&](Resource resource) -> bool {
          return !seen.insert(resource).second;
        });
    return Query(base(begin, std::distance(begin, end)));
  }

  /// Removes resources not of type @p U .
  template <class U>
  [[nodiscard]] auto ofType() -> Query<U> {
    const auto filter = where([](Resource resource) -> bool {
      return isa<U>(resource.getOperation());
    });
    return Query<U>(MutableArrayRef<U>(reinterpret_cast<U*>(filter.begin()),
                                       filter.size()));
  }

  /// Removes resources without a @p name attribute.
  [[nodiscard]] auto withAttribute(StringRef name) -> Query {
    if (base::empty()) {
      return *this;
    }

    return withAttribute(StringAttr::get(base::front()->getContext(), name));
  }
  /// Removes resources without a @p name attribute.
  [[nodiscard]] auto withAttribute(StringAttr name) -> Query {
    return where(
        [=](Resource resource) -> bool { return resource->hasAttr(name); });
  }
  /// Removes resources without an @p attr .
  [[nodiscard]] auto withAttribute(NamedAttribute attr) -> Query {
    return where([=](Resource resource) -> bool {
      return resource->getAttr(attr.getName()) == attr.getValue();
    });
  }

  /// Removes resources without a @p PropertyAttr .
  template <class PropertyAttr>
  [[nodiscard]] auto withProperty() -> Query {
    if (base::empty()) {
      return *this;
    }
    return withAttribute(
        PropertyAttr::getAttrName(base::front()->getContext()));
  }
  /// Removes resources without a @p property .
  template <class PropertyAttr>
  [[nodiscard]] auto withProperty(PropertyAttr property) -> Query {
    return withAttribute(NamedAttribute(property.getAttrName(), property));
  }

  /// Removes resources without a @p required feature.
  [[nodiscard]] auto withFeature(Feature required) -> Query {
    return where([=](Resource resource) -> bool {
      return resource.getFeature(required).has_value();
    });
  }
  /// Removes resources without the @p name feature.
  [[nodiscard]] auto withFeature(StringRef name) -> Query {
    if (base::empty()) {
      return *this;
    }

    return withFeature(StringAttr::get(base::front()->getContext(), name));
  }
  /// Removes resources without the @p name feature.
  [[nodiscard]] auto withFeature(StringAttr name) -> Query {
    return where([=](Resource resource) -> bool {
      return resource.getFeature(name).has_value();
    });
  }
  /// Removes resources without the @p FeatureAttr .
  template <class FeatureAttr>
  [[nodiscard]] auto withFeature() -> Query {
    if (base::empty()) {
      return *this;
    }
    return withFeature(FeatureAttr::getAttrName(base::front()->getContext()));
  }
  /// Removes resources without the @p required feature.
  template <class FeatureAttr>
  [[nodiscard]] auto withFeature(FeatureAttr required)
      -> std::enable_if_t<std::is_base_of_v<Attribute, FeatureAttr>, Query> {
    return where([=](Resource resource) -> bool {
      return resource.getFeature(required) != nullptr;
    });
  }

  /// Returns the single matching resource, if any.
  [[nodiscard]] auto singular() -> ResourceType {
    auto it = base::begin();
    const auto end = base::end();
    if (it == end) {
      return nullptr;
    }

    auto result = *it;
    for (++it; it != end; ++it) {
      if (*it != result) {
        return nullptr;
      }
    }

    return result;
  }

  /// Gets a value indicating whether any resource is matched.
  explicit operator bool() const { return !base::empty(); }
};

template <class T>
Query(MutableArrayRef<T>) -> Query<T>;
template <class T>
Query(llvm::SmallVectorImpl<T>) -> Query<T>;

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_QUERY_H_
