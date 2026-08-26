//===-- KTDF.cpp ------------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// PipelinePrivatizer
//===----------------------------------------------------------------------===//

namespace {

void erasePrivateOp(RewriterBase& rewriter, PrivateOp op, Block& result) {
  auto yield = op.getYieldOp();

  // Redirect all results to the yielded values for now.
  // NOTE: This might temporarily break the SSA property, which we fix later.
  rewriter.replaceAllOpUsesWith(op, yield->getOperands());

  // Inline the body of the old PrivateOp into the new body and erase it.
  rewriter.eraseOp(yield);
  // NOTE: We don't use the rewriter here, we notfify once during destroy.
  result.getOperations().splice(result.begin(), op.getBody()->getOperations());
  rewriter.eraseOp(op);
}

void inlineUnlinkedBlock(RewriterBase& rewriter, Block& source, Block& dest,
                         Block::iterator at) {
  if (auto* listener = rewriter.getListener(); listener) {
    while (!source.empty()) {
      rewriter.moveOpBefore(&source.front(), &dest, at);
    }

    return;
  }

  dest.getOperations().splice(at, source.getOperations());
}

}  // namespace

PipelinePrivatizer::PipelinePrivatizer(RewriterBase& rewriter,
                                       PipelineOp pipeline, bool force_recreate)
    : rewriter_(rewriter),
      pipeline_(pipeline),
      existing_(pipeline.getPrivateOp()) {
  if (force_recreate && existing_) {
    // Erase the existing PrivateOp.
    erasePrivateOp(rewriter_, existing_, private_);
    existing_ = nullptr;
  }
}

PipelinePrivatizer::~PipelinePrivatizer() {
  if (private_.empty()) {
    // Nothing was privated.
    return;
  }

  // Erase the existing PrivateOp, if any.
  if (existing_) {
    erasePrivateOp(rewriter_, existing_, private_);
  }

  // Collect the values that need to be yielded from the new PrivateOp.
  // This will re-discover the old results, since we redirected them.
  SmallVector<Value> yield_values;
  const auto should_yield = [&](Value value) -> bool {
    return value.isUsedOutsideOfBlock(&private_);
  };
  for (auto& op : private_) {
    llvm::append_range(yield_values,
                       llvm::make_filter_range(op.getResults(), should_yield));
  }

  // Create the new PrivateOp.
  OpBuilder::InsertionGuard guard(rewriter_);
  rewriter_.setInsertionPointToStart(pipeline_.getBody());
  auto target = mlir::ktdf::PrivateOp::create(
      rewriter_, pipeline_->getLoc(), TypeRange(yield_values),
      [&](OpBuilder& builder, Location loc) {
        mlir::ktdf::PrivateYieldOp::create(builder, loc, yield_values);
        inlineUnlinkedBlock(rewriter_, private_, *builder.getBlock(),
                            builder.getBlock()->begin());
      });

  // Redirect all uses of the private values outside of the PrivateOp.
  const auto is_outside_private = [&](OpOperand& use) -> bool {
    return !target.getBodyRegion().isAncestor(
        use.getOwner()->getParentRegion());
  };
  rewriter_.replaceUsesWithIf(yield_values, target->getResults(),
                              is_outside_private);
}

auto PipelinePrivatizer::isPrivate(Block* block) -> bool {
  while (block && block != existing_.getBody()) {
    auto* const parent = block->getParentOp();
    if (!parent) {
      break;
    }

    block = parent->getBlock();
  }

  return block == &private_;
}

auto PipelinePrivatizer::makePrivate(Operation* op) -> LogicalResult {
  if (isPrivate(op)) {
    // Op is already private.
    return success();
  }

  if (pipeline_->isProperAncestor(op)) {
    // Scan the parents of op to determine if it can be hoisted.
    for (auto* parent = op->getParentOp(); parent != pipeline_;
         parent = parent->getParentOp()) {
      if (parent->mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
        // Can't hoist op out of this parent.
        return failure();
      }
    }

    // Scan the operands of op to determine if it can be hoisted.
    for (auto opd : op->getOperands()) {
      if (isPrivate(opd.getParentBlock())) {
        // Source is already private, operands are fixed on destruction.
        continue;
      }
      if (auto parent = op->getParentRegion();
          parent && parent->isProperAncestor(&pipeline_.getBodyRegion())) {
        // Source is outside of the pipeline, so it can be hoisted.
        continue;
      }

      return failure();
    }
  } else {
    // Scan the results of op to determine if it can be sunk.
    const auto is_outside_pipeline = [&](Operation* user) -> bool {
      return !pipeline_->isProperAncestor(user);
    };
    if (llvm::any_of(op->getUsers(), is_outside_pipeline)) {
      // Can't sink into the pipeline.
      return failure();
    }
  }

  // Move it to the end of the new private body.
  // NOTE: We don't use the rewriter here, we notify once during destroy.
  op->moveBefore(&private_, private_.end());
  return success();
}
