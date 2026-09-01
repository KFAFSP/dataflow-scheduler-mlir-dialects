//===-- InstantiateNeighborhoods.cpp ----------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceIds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h"  // IWYU pragma: keep

#define DEBUG_TYPE "ktdfarch-instantiate-neighborhoods"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace mlir::ktdf_arch {
#define GEN_PASS_DEF_INSTANTIATENEIGHBORHOODSPASS
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf_arch

namespace {

/// Instantiates copies of neighborhoods according to their domain.
struct InstantiateNeighborhood : OpRewritePattern<NeighborhoodOp> {
  InstantiateNeighborhood(MLIRContext* context, ResourceIds& resource_ids)
      : OpRewritePattern(context), resource_ids_(resource_ids) {}

  auto matchAndRewrite(NeighborhoodOp source, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!source.isLeaf()) {
      return rewriter.notifyMatchFailure(source, "neighborhood is not a leaf");
    }
    if (source.isSingleton()) {
      return rewriter.notifyMatchFailure(source, "neighborhood is a singleton");
    }
    SmallVector<NeighborOp> users;
    for (auto* user : source->getUsers()) {
      auto neighbor = dyn_cast<NeighborOp>(user);
      if (!neighbor) {
        return rewriter.notifyMatchFailure(source,
                                           "neighborhood has unknown user");
      }

      users.push_back(neighbor);
    }
    for (auto* user : source.getSelfArgument().getUsers()) {
      if (!isa<NeighborOp>(user)) {
        return rewriter.notifyMatchFailure(
            source, "neighborhood has unknown recursive user");
      }

      // These aren't collected since they will be cloned anyways.
    }

    instantiate(rewriter, source, users);
    return success();
  }

 private:
  static void inlinePoint(RewriterBase& rewriter, NeighborOp target,
                          ArrayRef<int64_t> point) {
    auto map = target.getMap();
    assert(map.getNumDims() >= point.size());

    // Substitute in the point for the trailing dims, removing them.
    SmallVector<AffineExpr> substitutions;
    substitutions.reserve(map.getNumDims());
    for (auto i = 0U; i < map.getNumDims() - point.size(); ++i) {
      substitutions.push_back(getAffineDimExpr(i, target.getContext()));
    }
    for (auto i : point) {
      substitutions.push_back(getAffineConstantExpr(i, target.getContext()));
    }
    map = map.replaceDimsAndSymbols(substitutions, {},
                                    map.getNumDims() - point.size(), 0);
    rewriter.modifyOpInPlace(target, [&]() { target.setMap(map); });
  }
  static void inlinePoint(RewriterBase& rewriter, Block& target,
                          ArrayRef<int64_t> point) {
    target.walk([&](NeighborOp op) { inlinePoint(rewriter, op, point); });
  }

  static auto flatten(OpBuilder& builder, ArrayRef<AffineExpr> exprs,
                      ArrayRef<int64_t> domain) -> AffineExpr {
    assert(exprs.size() == domain.size());

    auto result = exprs.front();
    for (auto i = 1U; i < domain.size(); ++i) {
      result = result * builder.getAffineConstantExpr(domain[i]) + exprs[i];
    }

    return result;
  }
  static void flatten(RewriterBase& rewriter, NeighborOp source,
                      ArrayRef<int64_t> domain, ValueRange targets) {
    auto map = source.getMap();
    assert(map.getNumResults() >= domain.size());
    assert(!domain.empty());

    // Replace the leading result dimensions with the flattened result.
    auto results = llvm::to_vector(map.getResults());
    results.front() = flatten(
        rewriter, ArrayRef<AffineExpr>(results).take_front(domain.size()),
        domain);
    results.erase(results.begin(),
                  std::next(results.begin(), domain.size() - 1));
    map = AffineMap::get(map.getNumDims(), 0, results, rewriter.getContext());
    rewriter.modifyOpInPlace(source, [&]() {
      source.setMap(map);
      source->setOperands(targets);
    });
  }

  void instantiate(RewriterBase& rewriter, NeighborhoodOp source,
                   SmallVectorImpl<NeighborOp>& users) const {
    const auto domain = source.getNeighborhoodType().getDimensions();

    // Prepare an iterator over the points in the domain.
    SmallVector<int64_t> point(domain.size(), 0U);
    const auto next = [&]() -> bool {
      for (auto [it, dim] = std::pair(point.rbegin(), domain.rbegin());
           it != point.rend(); ++it, ++dim) {
        if (++*it >= *dim) {
          *it = 0;
          continue;
        }

        return true;
      }

      return false;
    };

    // Create a trivial neighborhood per point, substituting the point into
    // every nested NeighborOp (self references still target this instance).
    const auto target_type = source.getNeighborhoodType().cloneWith(
        std::nullopt, ArrayRef<int64_t>{});
    SmallVector<Value> targets;
    do {
      const auto target_builder = [&](OpBuilder& builder, Location,
                                      Value self) {
        IRMapping mapping;
        mapping.map(source.getSelfArgument(), self);
        for (auto& op : *source.getBody()) {
          builder.clone(op, mapping);
        }
        inlinePoint(rewriter, *builder.getBlock(), point);

        // Add all the cloned self users to the list of users.
        for (auto* const user : self.getUsers()) {
          // NOTE: The pattern was rejected if any user isn't a NeighborOp.
          users.push_back(cast<NeighborOp>(user));
        }
      };
      auto target = NeighborhoodOp::create(rewriter, source.getLoc(),
                                           target_type, target_builder);
      targets.push_back(target);

      // Fix up all the resource IDs that were duplicated.
      target.walk([&](Resource resource) {
        if (const auto maybe_id = resource.removeIdAttr(); maybe_id) {
          resource_ids_.getOrAssign(resource, cast<StringAttr>(maybe_id));
        }
      });
    } while (next());

    // Flatten all users to the unrolled targets, then erase the source.
    for (auto user : users) {
      flatten(rewriter, user, domain, targets);
    }
    source.walk([&](Resource resource) {
      if (resource.getId()) {
        resource_ids_.assign(resource, StringAttr{});
      }
    });
    rewriter.eraseOp(source);
  }

  ResourceIds& resource_ids_;
};

}  // namespace

//===----------------------------------------------------------------------===//
// InstantiateNeighborhoodsPass
//===----------------------------------------------------------------------===//

namespace {

struct InstantiateNeighborhoodsPass
    : public ktdf_arch::impl::InstantiateNeighborhoodsPassBase<
          InstantiateNeighborhoodsPass> {
  using InstantiateNeighborhoodsPassBase::InstantiateNeighborhoodsPassBase;

  void runOnOperation() override {
    DeviceRef device(getOperation(), getAnalysisManager());
    auto& resource_ids = device.getOrCreateView<ResourceIds>();

    RewritePatternSet patterns(&getContext());
    patterns.add<InstantiateNeighborhood>(&getContext(), resource_ids);
    NeighborhoodOp::getCanonicalizationPatterns(patterns,
                                                patterns.getContext());
    NeighborOp::getCanonicalizationPatterns(patterns, patterns.getContext());

    auto changed = false;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     GreedyRewriteConfig(), &changed))) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](NeighborhoodOp op) {
      op->emitError("unable to instantiate neighborhood");
      signalPassFailure();
    });

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

}  // namespace
