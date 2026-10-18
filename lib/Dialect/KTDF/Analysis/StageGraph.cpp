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

ConsumerEdgeIterator::ConsumerEdgeIterator(StageOp stage) {
  if (auto deps = stage.getDependsOutMutable(); !deps.empty()) {
    operand_ = deps.begin();
    use_ = operand_->get().use_begin();
    seek();
  }
}

auto ConsumerEdgeIterator::operator++() -> ConsumerEdgeIterator& {
  assert(operand_);
  ++use_;
  seek();
  return *this;
}

void ConsumerEdgeIterator::seek() {
  auto* const end =
      cast<StageOp>(operand_->getOwner()).getDependsOutMutable().end();
  while (true) {
    for (const auto use_end = operand_->get().use_end(); use_ != use_end;
         ++use_) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isInDependency(*use_)) {
        return;
      }
    }

    if (++operand_ == end) {
      break;
    }
    use_ = operand_->get().use_begin();
  }

  *this = ConsumerEdgeIterator();
}

//===----------------------------------------------------------------------===//
// ProducerIterator
//===----------------------------------------------------------------------===//

ProducerEdgeIterator::ProducerEdgeIterator(StageOp stage) {
  if (auto deps = stage.getDependsInMutable(); !deps.empty()) {
    operand_ = deps.begin();
    use_ = operand_->get().use_begin();
    seek();
  }
}

auto ProducerEdgeIterator::operator++() -> ProducerEdgeIterator& {
  assert(operand_);
  ++use_;
  seek();
  return *this;
}

void ProducerEdgeIterator::seek() {
  auto* const end =
      cast<StageOp>(operand_->getOwner()).getDependsInMutable().end();
  while (true) {
    for (const auto use_end = operand_->get().use_end(); use_ != use_end;
         ++use_) {
      if (auto stage = dyn_cast<StageOp>(use_->getOwner());
          stage && stage.isOutDependency(*use_)) {
        return;
      }
    }

    if (++operand_ == end) {
      break;
    }
    use_ = operand_->get().use_begin();
  }

  *this = ProducerEdgeIterator();
}

//===----------------------------------------------------------------------===//
// StageGraph::Node
//===----------------------------------------------------------------------===//

auto StageGraph::Node::dependsOn(StageOp producer) const -> bool {
  return llvm::is_contained(llvm::breadth_first(llvm::Inverse<Node>(*this)),
                            producer);
}
