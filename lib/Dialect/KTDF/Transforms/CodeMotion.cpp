//===-- CodeMotion.cpp ------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/CodeMotion.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// hoistPipelineContents
//===----------------------------------------------------------------------===//

auto mlir::ktdf::hoistPipelineContents(
    PipelineOp pipeline, function_ref<PipelineAnchor(Operation*)> get_anchor,
    function_ref<void(Operation*)> move_outside_pipeline) -> size_t {
  const auto is_defined_outside = [&](Value value) -> bool {
    return value.getParentRegion()->isProperAncestor(&pipeline.getBodyRegion());
  };
  const auto can_move_outside = [&](Operation* op) {
    return llvm::all_of(op->getOperands(), is_defined_outside);
  };
  const auto default_move = [&](Operation* op) { op->moveBefore(pipeline); };
  if (!move_outside_pipeline) {
    move_outside_pipeline = default_move;
  }

  size_t result = 0U;

  // Visit the stages first.
  {
    IRRewriter rewriter(pipeline);
    PipelinePrivatizer privatizer(rewriter, pipeline);

    for (auto stage : pipeline.getStages()) {
      for (auto& op : llvm::make_early_inc_range(stage.getOps())) {
        // Skip all ops that can't leave the stage.
        if (llvm::any_of(op.getOperands(), [&](Value operand) -> bool {
              return stage.getBodyRegion().isAncestor(
                  operand.getParentRegion());
            })) {
          continue;
        }

        switch (get_anchor(&op)) {
          case PipelineAnchor::Stage:
            // Op should stay where it is.
            break;
          case PipelineAnchor::Private:
            // If possible, make the op private.
            if (succeeded(privatizer.makePrivate(&op))) {
              ++result;
            }
            break;
          case PipelineAnchor::Parent:
            // If possible, move the op outside of the pipeline.
            if (can_move_outside(&op)) {
              move_outside_pipeline(&op);
              ++result;
            }
            break;
        }
      }
    }
  }

  // Visit the PrivateOp last, if there is any.
  if (auto priv = pipeline.getPrivateOp(); priv) {
    for (auto& op : llvm::make_early_inc_range(priv.getOps())) {
      if (isa<CreateTokenOp>(op)) {
        // Although they are pure, tokens may never be hoisted out of PrivateOp.
        continue;
      }

      if (!can_move_outside(&op) || get_anchor(&op) != PipelineAnchor::Parent) {
        // Op should stay where it is.
        continue;
      }

      move_outside_pipeline(&op);
      ++result;
    }
  }

  return result;
}
