//===-- CodeMotion.h --------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_CODEMOTION_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_CODEMOTION_H_

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// Enumeration of different locations an op is anchored w.r.t. a PipelineOp.
enum class PipelineAnchor : char {
  /// Operation is anchored within a stage.
  Stage = 0,
  /// Operation is a private resource shared among stages.
  Private,
  /// Operation is a shared resource outside the pipeline.
  Parent
};

/// Hoist the contents of a @p pipeline .
///
/// This function will visit all immediate children of all stages and any
/// PrivateOp of the @p pipeline and call @p get_anchor to determine where the
/// operation should be placed. If possible, it will then hoist the op to that
/// requested level, using @p move_outside_pipeline to hoist above the pipeline.
///
/// If @p move_outside_pipeline is `nullptr`, the default move strategy is used.
///
/// @return Number of operations hoisted.
auto hoistPipelineContents(
    PipelineOp pipeline, function_ref<PipelineAnchor(Operation*)> get_anchor,
    function_ref<void(Operation*)> move_outside_pipeline = nullptr) -> size_t;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_CODEMOTION_H_
