//===-- Links.h -------------------------------------------------*- c++ -*-===//
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
// This file declares helpers for Link-based queries of the architecture graph.
//
// Every result returned from a leaf Resource is an Endpoint of that resource,
// which may be used by a Link operation to establish a link. However, not every
// value in the architecture graph is an Endpoint, since GroupOp instance can
// break up def-use-chains to enforce private resource boundaries.
//
// The Endpoint type and its getEndpoint/getResource function implement the
// task of traversing the def-use-chain until the actual declaration is found.
// This is used by the visitLinks and getLink* utilities to then look for
// instances of Link ops acting on them.
//
// TODO: Caching of Value->Endpoint mappings in an anlaysis that simplifies
//       link lookup might be helpful in the future.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_LINKS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_LINKS_H_

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"

namespace mlir::ktdf_arch {

//===----------------------------------------------------------------------===//
// Endpoints
//===----------------------------------------------------------------------===//

/// Represents an endpoint of a leaf Resource.
class Endpoint : public OpResult {
 public:
  [[nodiscard]] static auto classof(Value value) -> bool {
    const auto result = dyn_cast<OpResult>(value);
    return result && classof(result);
  }
  [[nodiscard]] static auto classof(OpResult result) -> bool {
    return llvm::isa<Resource>(result.getOwner()) &&
           !llvm::isa<GroupOp>(result.getOwner());
  }

  using OpResult::OpResult;

  [[nodiscard]] auto getOwner() -> Resource {
    return cast<Resource>(OpResult::getOwner());
  }
};

/// Tries to obtain the Endpoint defining @p value .
[[nodiscard]] auto getEndpoint(Value value) -> Endpoint;

/// Tries to obtain the Resource of @p ResourceType defining @p value .
///
/// @retval nullptr       @p value is not an endpoint of a @p ResourceType .
/// @retval ResourceType  Resource defining @p value .
template <class ResourceType = Resource>
[[nodiscard]] auto getResource(Value value) -> ResourceType {
  static_assert(
      std::is_same_v<ResourceType, Resource> ||
      std::is_base_of_v<detail::ResourceTrait<ResourceType>, ResourceType>);

  if (auto endpoint = getEndpoint(value); endpoint) {
    return dyn_cast<ResourceType>(endpoint.getOwner().getOperation());
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Links
//===----------------------------------------------------------------------===//

/// Indicates a Resource-relative direction in which a link is going.
enum class LinkDirection : char {
  /// Link is incoming to the resource.
  Incoming = 0b01,
  /// Link is outgoing from the resource.
  Outgoing = 0b10,
  /// Link is both incoming and outgoing from the resource.
  Bidirectional = 0b11,

  LLVM_MARK_AS_BITMASK_ENUM(Outgoing)
};

/// Determines whether @p direction has an outgoing component.
[[nodiscard]] inline auto isOutgoing(LinkDirection direction) -> bool {
  return (direction & LinkDirection::Outgoing) == LinkDirection::Outgoing;
}
/// Determines whether @p direction has an incoming component.
[[nodiscard]] inline auto isIncoming(LinkDirection direction) -> bool {
  return (direction & LinkDirection::Incoming) == LinkDirection::Incoming;
}

namespace detail {

auto visitLinks(Endpoint endpoint,
                function_ref<bool(Link, LinkDirection)> callback) -> bool;

template <class Callback>
[[nodiscard]] auto makeLinkVisitor(Callback&& callback) -> decltype(auto) {
  using traits = llvm::function_traits<std::remove_reference_t<Callback>>;
  using result_type = typename traits::result_t;
  constexpr auto is_void = std::is_void_v<result_type>;
  static_assert(is_void || std::is_same_v<bool, result_type>);
  static_assert(traits::num_args >= 1);
  using link_type = typename traits::template arg_t<0>;

  constexpr auto make_invoke = [](auto&& callback) {
    static_assert(traits::num_args >= 1 && traits::num_args <= 2);
    if constexpr (traits::num_args == 1) {
      return [callback = std::forward<decltype(callback)>(callback)](
                 link_type link, LinkDirection /*direction*/) -> bool {
        if constexpr (is_void) {
          callback(link);
          return true;
        } else {
          return callback(link);
        }
      };
    } else /*if constexpr (traits::num_args == 2)*/ {
      using direction_type = typename traits::template arg_t<1>;
      static_assert(std::is_convertible_v<LinkDirection, direction_type>);
      return [callback = std::forward<decltype(callback)>(callback)](
                 link_type link, LinkDirection direction) -> bool {
        if constexpr (is_void) {
          callback(link, direction);
          return true;
        } else {
          return callback(link, direction);
        }
      };
    }
  };

  if constexpr (std::is_same_v<link_type, Link>) {
    if constexpr (traits::num_args == 1) {
      return make_invoke(std::forward<Callback>(callback));
    } else {
      return std::forward<Callback>(callback);
    }
  } else {
    static_assert(std::is_base_of_v<LinkTrait<link_type>, link_type>);
    return [invoke = make_invoke(std::forward<Callback>(callback))](
               Link link, LinkDirection direction) -> bool {
      auto of_type = dyn_cast<link_type>(link.getOperation());
      if (!of_type) {
        return true;
      }
      return invoke(of_type, direction);
    };
  }
}

}  // namespace detail

/// Visits all Links attached to @p endpoint .
///
/// The @p callback must have a first argument that is either Link or a concrete
/// op type that is statically known to implement the Link interface, i.e.,
/// has the associated trait. The @p callback may have a second parameter that
/// accepts a LinkDirection, indicating the directionality of the link.
///
/// The @p callback must be an invocable returning either `void` or `bool`. If
/// `false` is returned, the traversal is aborted and `false` is propagated,
/// otherwise the result is `true`/`void`.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Endpoint endpoint, Callback&& callback) -> ReturnType {
  const auto result = detail::visitLinks(
      endpoint, detail::makeLinkVisitor(std::forward<Callback>(callback)));
  if constexpr (!std::is_void_v<ReturnType>) {
    return result;
  }
}
/// Visits all Links attached to @p resource .
///
/// See visitLinks(Endpoint, auto&&) for more information on the callback.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Resource resource, Callback&& callback) -> ReturnType {
  if (isa<GroupOp>(resource)) {
    if constexpr (!std::is_void_v<ReturnType>) {
      return true;
    } else {
      return;
    }
  }
  return llvm::all_of(
      resource->getResults(),
      [invoke = detail::makeLinkVisitor(std::forward<Callback>(callback))](
          OpResult endpoint) -> bool {
        return detail::visitLinks(cast<Endpoint>(endpoint), invoke);
      });
}
/// Visits all Links from @p source to @p target .
///
/// See visitLinks(Endpoint, auto&&) for more information on the callback.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Endpoint source, Endpoint target, Callback&& callback)
    -> ReturnType {
  return visitLinks(
      source,
      [invoke = detail::makeLinkVisitor(std::forward<Callback>(callback)),
       target = target](Link link, LinkDirection direction) -> bool {
        if (!isOutgoing(direction)) {
          return true;
        }
        for (auto value : link.getTargets()) {
          if (getEndpoint(value) == target) {
            return invoke(link, direction);
          }
        }
        return true;
      });
}
/// Visits all Links from @p source to @p target .
///
/// See visitLinks(Endpoint, auto&&) for more information on the callback.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Resource source, Resource target, Callback&& callback)
    -> ReturnType {
  return visitLinks(
      source,
      [invoke = detail::makeLinkVisitor(std::forward<Callback>(callback)),
       target = target](Link link, LinkDirection direction) -> bool {
        if (!isOutgoing(direction)) {
          return true;
        }
        for (auto value : link.getTargets()) {
          if (getResource(value) == target) {
            return invoke(link, direction);
          }
        }
        return true;
      });
}

/// Collects Links in @p result .
///
/// The first argument may be convertible to LinkDirection, in which case it
/// filters the direction of the collected links. All other arguments are
/// forwarded to visitLinks().
template <class LinkType = Link, class Head, class... Tail>
void getLinks(SmallVectorImpl<LinkType>& result, Head&& head, Tail&&... tail) {
  if constexpr (std::is_convertible_v<Head, LinkDirection>) {
    visitLinks(
        std::forward<Tail>(tail)...,
        [&, filter = head](LinkType link, LinkDirection direction) -> bool {
          if ((direction & filter) == filter) {
            result.push_back(link);
          }
          return true;
        });
  } else {
    visitLinks(std::forward<Head>(head), std::forward<Tail>(tail)...,
               [&](LinkType link, LinkDirection) -> bool {
                 result.push_back(link);
                 return true;
               });
  }
}

/// Collects Links.
///
/// See getLinks(SmallVectorImpl<LinkType>&, auto&&, auto&&...) for more info.
template <class LinkType = Link, class Head, class... Tail>
[[nodiscard]] auto getLinks(Head&& head, Tail&&... tail)
    -> std::enable_if_t<!std::is_base_of_v<llvm::SmallVectorImpl<LinkType>,
                                           std::remove_reference_t<Head>>,
                        SmallVector<LinkType>> {
  SmallVector<LinkType> result;
  getLinks(result, std::forward<Head>(head), std::forward<Tail>(tail)...);
  return result;
}

/// Finds a unique Link.
///
/// The first argument may be convertible to LinkDirection, in which case it
/// filters the direction of the link searched for. All other arguments are
/// forwarded to visitLinks().
template <class LinkType = Link, class Head, class... Tail>
[[nodiscard]] auto getLink(Head&& head, Tail&&... tail) -> LinkType {
  LinkType result;
  bool unique;
  if constexpr (std::is_convertible_v<Head, LinkDirection>) {
    unique = visitLinks(
        std::forward<Tail>(tail)...,
        [&, filter = head](LinkType link, LinkDirection direction) -> bool {
          if ((direction & filter) != filter) {
            return true;
          }
          if (result) {
            return false;
          }
          result = link;
          return true;
        });
  } else {
    unique = visitLinks(std::forward<Head>(head), std::forward<Tail>(tail)...,
                        [&](LinkType link, LinkDirection) -> bool {
                          if (result) {
                            return false;
                          }
                          result = link;
                          return true;
                        });
  }
  return unique ? result : nullptr;
}

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_LINKS_H_
