//===-- StageGraph.cpp ------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"

#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// ConsumerIterator
//===----------------------------------------------------------------------===//

ConsumerEdgeIterator::ConsumerEdgeIterator(StageOp stage)
    : operand_(stage.getDependsOutMutable().begin()) {
  if (operand_ != stage.getDependsOutMutable().end()) {
    const auto end = operand_->get().use_end();
    use_ = operand_->get().use_begin();
    if (use_ != end) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isInDependency(*use_)) {
        return;
      }
    }

    ++*this;
  }
}

auto ConsumerEdgeIterator::operator++() -> ConsumerEdgeIterator& {
  const auto advance_use = [&]() -> bool {
    const auto end = use_->get().use_end();
    while (++use_ != end) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isInDependency(*use_)) {
        return true;
      }
    }

    return false;
  };

  if (!advance_use()) {
    auto* const end =
        cast<StageOp>(operand_->getOwner()).getDependsOutMutable().end();
    while (++operand_ != end) {
      use_ = operand_->get().use_begin();
      if (advance_use()) {
        break;
      }
    }
    use_ = {};
  }

  return *this;
}

//===----------------------------------------------------------------------===//
// ProducerIterator
//===----------------------------------------------------------------------===//

ProducerEdgeIterator::ProducerEdgeIterator(StageOp stage)
    : operand_(stage.getDependsInMutable().begin()) {
  if (operand_ != stage.getDependsInMutable().end()) {
    const auto end = operand_->get().use_end();
    use_ = operand_->get().use_begin();
    if (use_ != end) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isOutDependency(*use_)) {
        return;
      }
    }

    ++*this;
  }
}

auto ProducerEdgeIterator::operator++() -> ProducerEdgeIterator& {
  const auto advance_use = [&]() -> bool {
    const auto end = use_->get().use_end();
    while (++use_ != end) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isOutDependency(*use_)) {
        return true;
      }
    }

    return false;
  };

  if (!advance_use()) {
    auto* const end =
        cast<StageOp>(operand_->getOwner()).getDependsInMutable().end();
    while (++operand_ != end) {
      use_ = operand_->get().use_begin();
      if (advance_use()) {
        break;
      }
    }
    use_ = {};
  }

  return *this;
}

//===----------------------------------------------------------------------===//
// StageGraph::Node
//===----------------------------------------------------------------------===//

auto StageGraph::Node::dependsOn(StageOp producer) const -> bool {
  return llvm::is_contained(llvm::breadth_first(llvm::Inverse<Node>(*this)),
                            producer);
}
