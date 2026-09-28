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

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/WalkResult.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// PrivateBuilder
//===----------------------------------------------------------------------===//

PrivateBuilder::PrivateBuilder(PrivateOp existing,
                               OpBuilder::Listener* listener)
    : ImplicitLocOpBuilder(existing.getLoc(), existing.getContext(), listener) {
  OpBuilder::setInsertionPoint(existing.getBody()->getTerminator());
}

PrivateBuilder::PrivateBuilder(PipelineOp pipeline, std::optional<Location> loc,
                               OpBuilder::Listener* listener)
    : ImplicitLocOpBuilder(loc.value_or(pipeline.getLoc()),
                           pipeline->getContext(), listener) {
  auto existing = pipeline.getPrivateOp();
  if (!existing) {
    OpBuilder::setInsertionPointToStart(pipeline.getBody());
    existing =
        PrivateOp::create(*this, {}, [](OpBuilder& builder, Location loc) {
          PrivateYieldOp::create(builder, loc);
        });
  }

  OpBuilder::setInsertionPoint(existing.getBody()->getTerminator());
}

auto PrivateBuilder::canPrivate(Operation* op) const -> bool {
  auto private_op = cast<PrivateOp>(getInsertionBlock()->getParentOp());
  auto* const scope = private_op->getParentRegion();

  // Determine if all users of op can see the results of the PrivateOp.
  const auto sees_results = [&](Operation* user) -> bool {
    if (user->getParentRegion() == scope) {
      return true;
    }
    for (user = user->getParentOp(); user; user = user->getParentOp()) {
      if (user->mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
        return false;
      }
      if (user->getParentRegion() == scope) {
        return true;
      }
    }

    return false;
  };
  if (!llvm::all_of(op->getUsers(), sees_results)) {
    return false;
  }

  // Determine if all (transitive) uses in op are available in the PrivateOp.
  const auto check_available = op->walk([&](Operation* op) -> WalkResult {
    const auto is_available = [&](Value value) {
      return value.getParentRegion()->isAncestor(&private_op.getBodyRegion());
    };
    if (!llvm::all_of(op->getOperands(), is_available)) {
      return WalkResult::interrupt();
    }
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return !check_available.wasInterrupted();
}

void PrivateBuilder::makePrivate(Operation* op) {
  assert(canPrivate(op));

  const InsertPoint from(op->getBlock(), Block::iterator(op));
  op->moveBefore(getInsertionBlock(), getInsertionPoint());
  getListener()->notifyOperationInserted(op, from);
}

auto PrivateBuilder::createToken(std::optional<Location> loc) -> Token {
  return CreateTokenOp::create(*this, loc.value_or(getLoc()));
}

auto PrivateBuilder::createFifo(ArrayRef<FifoSlotType> slots,
                                ValueRange dynamic_sizes,
                                std::optional<Location> loc) -> ValueRange {
  return FifoAllocateOp::create(*this, loc.value_or(getLoc()),
                                ArrayRef<Type>(slots.data(), slots.size()),
                                dynamic_sizes)
      .getResults();
}

auto PrivateBuilder::build() -> PrivateOp {
  IRRewriter rewriter(*this);
  auto* block = getInsertionBlock();

  // Redirect all yielded results back to their definitions, since we might
  // need to re-create the PrivateOp.
  auto yield = cast<PrivateYieldOp>(block->getTerminator());
  for (auto result : block->getParentOp()->getOpResults()) {
    rewriter.replaceAllUsesWith(result,
                                yield.getOperand(result.getResultNumber()));
  }

  // Collect the values that need to be yielded from the new PrivateOp.
  // This will re-discover the old results, since we redirected them.
  SmallVector<Value> yield_values;
  for (auto& op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    if (mlir::isOpTriviallyDead(&op)) {
      rewriter.eraseOp(&op);
      continue;
    }

    const auto should_yield = [&](Value value) -> bool {
      return value.isUsedOutsideOfBlock(block);
    };
    llvm::append_range(
        yield_values,
        llvm::make_filter_range(llvm::reverse(op.getResults()), should_yield));
  }
  std::reverse(yield_values.begin(), yield_values.end());

  auto result = cast<PrivateOp>(block->getParentOp());

  // Determine if we need to re-create the PrivateOp.
  const TypeRange yield_types(yield_values);
  if (result->getResultTypes() != yield_types) {
    // Insert a new PrivateOp just before the existing one.
    rewriter.setInsertionPoint(result);
    auto old_op = std::exchange(
        result, PrivateOp::create(rewriter, result.getLoc(), yield_types));

    // Copy over the attributes and move the body.
    result->setDiscardableAttrs(old_op->getRawDictionaryAttrs());
    rewriter.inlineBlockBefore(block, result.getBody(),
                               result.getBody()->end());
    block = result.getBody();

    // The terminator was moved, and the old op must now be deleted.
    OpBuilder::setInsertionPoint(yield);
    rewriter.eraseOp(old_op);
  }

  // Update the terminator and redirect pipeline uses of private values.
  rewriter.modifyOpInPlace(yield, [&]() { yield->setOperands(yield_values); });
  const auto is_outside_private = [&](OpOperand& use) -> bool {
    return !result.getBodyRegion().isAncestor(
        use.getOwner()->getParentRegion());
  };
  rewriter.replaceUsesWithIf(yield_values, result.getResults(),
                             is_outside_private);
  return result;
}
