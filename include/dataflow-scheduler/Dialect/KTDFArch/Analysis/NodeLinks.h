//===-- NodeLinks.h ---------------------------------------------*- c++ -*-===//
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
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODELINKS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODELINKS_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Pass/AnalysisManager.h>

#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/NodeEndpoints.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"

namespace mlir::ktdf_arch {

/// Indicates a Node-relative direction in which a link is going.
enum class LinkDirection : char {
  /// Link is incoming to the node.
  Incoming = 0b01,
  /// Link is outgoing from the node.
  Outgoing = 0b10,
  /// Link is both incoming and outgoing from the node.
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
///
/// Consider using a LinkCollection cached by NodeLinks instead.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Endpoint endpoint, Callback&& callback) -> ReturnType {
  const auto result = detail::visitLinks(
      endpoint, detail::makeLinkVisitor(std::forward<Callback>(callback)));
  if constexpr (!std::is_void_v<ReturnType>) {
    return result;
  }
}
/// Visits all Links attached to @p node .
///
/// See visitLinks(Endpoint, auto&&) for more information on the callback.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Node node, Callback&& callback) -> ReturnType {
  return llvm::all_of(
      node->getResults(),
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
auto visitLinks(Endpoint source, Endpoint target, NodeEndpoints& endpoints,
                Callback&& callback) -> ReturnType {
  return visitLinks(
      source,
      [invoke = detail::makeLinkVisitor(std::forward<Callback>(callback)),
       target = target,
       &endpoints](Link link, LinkDirection direction) -> bool {
        if (!isOutgoing(direction)) {
          return true;
        }
        for (auto value : link.getTargets()) {
          if (endpoints[value] == target) {
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
auto visitLinks(Endpoint source, Endpoint target, Callback&& callback)
    -> ReturnType {
  NodeEndpoints endpoints;
  return visitLinks(source, target, endpoints,
                    std::forward<Callback>(callback));
}
/// Visits all Links from @p source to @p target .
///
/// See visitLinks(Endpoint, auto&&) for more information on the callback.
template <class Callback, class ReturnType = typename llvm::function_traits<
                              std::remove_reference_t<Callback>>::result_t>
auto visitLinks(Node source, Node target, NodeEndpoints& endpoints,
                Callback&& callback) -> ReturnType {
  return visitLinks(
      source,
      [invoke = detail::makeLinkVisitor(std::forward<Callback>(callback)),
       target = target,
       &endpoints](Link link, LinkDirection direction) -> bool {
        if (!isOutgoing(direction)) {
          return true;
        }
        for (auto value : link.getTargets()) {
          if (endpoints.getNode(value) == target) {
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
auto visitLinks(Node source, Node target, Callback&& callback) -> ReturnType {
  NodeEndpoints endpoints;
  return visitLinks(source, target, endpoints,
                    std::forward<Callback>(callback));
}

/// Collects Links in @p result .
///
/// The first argument may be convertible to LinkDirection, in which case it
/// filters the direction of the collected links. All other arguments are
/// forwarded to visitLinks().
///
/// Consider using a LinkCollection cached by NodeLinks instead.
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
///
/// Consider using a LinkCollection cached by NodeLinks instead.
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

//===----------------------------------------------------------------------===//
// LinkCollection
//===----------------------------------------------------------------------===//

/// Stores all the links of a Node.
///
/// Internally, the links are sorted by direction, such that incoming, outgoing
/// and bidirectional links are always contiguous ranges. For example:
///
///                      /--getBidirectional--\
///          /-----------getIncoming----------\
///        { in, in, in, in|out, in|out, in|out, out, out, out }
///                      \------------getOutgoing------------/
///
/// Consider using the cached NodeLinks analysis instead.
class LinkCollection {
  using storage_type = llvm::SmallVector<Link, 4>;

 public:
  /// Creates the LinkCollection of @p node .
  explicit LinkCollection(Node node);

  /// Gets the number of incoming links.
  [[nodiscard]] auto getNumIncoming() const -> unsigned {
    return size() - num_out_only_;
  }
  /// Gets the number of outgoing links.
  [[nodiscard]] auto getNumOutgoing() const -> unsigned {
    return size() - num_in_only_;
  }
  /// Gets the number of bidirectional links.
  [[nodiscard]] auto getNumBidirectional() const -> unsigned {
    return size() - num_in_only_ - num_out_only_;
  }

  /// Gets the incoming links.
  [[nodiscard]] auto getIncoming() const -> ArrayRef<Link> {
    return asArrayRef().take_front(num_in_only_);
  }
  /// Gets the outgoing links.
  [[nodiscard]] auto getOutgoing() const -> ArrayRef<Link> {
    return asArrayRef().take_back(num_out_only_);
  }
  /// Gets the bidirectional links.
  [[nodiscard]] auto getBidirectional() const -> ArrayRef<Link> {
    return asArrayRef().drop_front(num_in_only_).drop_back(num_out_only_);
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = Link;
  using size_type = storage_type::size_type;
  using iterator = storage_type::const_iterator;

  [[nodiscard]] auto empty() const -> bool { return links_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return links_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return links_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return links_.end(); }

  [[nodiscard]] auto asArrayRef() const -> ArrayRef<Link> { return links_; }

 private:
  unsigned num_in_only_ = 0;
  unsigned num_out_only_ = 0;
  storage_type links_;
};

//===----------------------------------------------------------------------===//
// NodeLinks
//===----------------------------------------------------------------------===//

/// Analysis that caches LinkCollections for Nodes.
class NodeLinks {
  using map_type = llvm::DenseMap<Node, LinkCollection>;

 public:
  /*implicit*/ NodeLinks() = default;
  explicit NodeLinks(mlir::Operation* /*op*/) {
    // This doesn't have to be a DeviceView, so we keep it a general analysis.
  }

  static auto isInvalidated(const AnalysisManager::PreservedAnalyses& /*pa*/)
      -> bool {
    // By default, devices are considered immutable.
    return false;
  }

  /// Gets the LinkCollection for @p node .
  [[nodiscard]] auto get(Node node) -> const LinkCollection& {
    return cache_.try_emplace(node, node).first->second;
  }
  /// @copydoc get(Node)
  [[nodiscard]] auto operator[](Node node) -> const LinkCollection& {
    return get(node);
  }

 private:
  map_type cache_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_NODELINKS_H_
