//===-- StageDependency.h ---------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEDEPENDENCY_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEDEPENDENCY_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// Cache that stores dependency edges between pipeline stages.
class StageDependency {
 public:
  /// Initializes a new StageDependency cache.
  /*implicit*/ StageDependency() = default;

  /// Gets the producers that @p consumer depends on.
  static void getProducers(StageOp consumer,
                           SmallPtrSetImpl<StageOp>& producers,
                           bool transitive = true);
  /// Gets the consumers that depend on @p producer .
  static void getConsumers(StageOp producer,
                           SmallPtrSetImpl<StageOp>& consumers,
                           bool transitive = true);

  /// Determines whether @p consumer depends on @p producer .
  auto contains(StageOp consumer, StageOp producer, bool transitive = false)
      -> bool;

  /// Updates the cache that @p consumer depends on @p producer .
  ///
  /// @warning  This does not modify the IR.
  auto insert(StageOp consumer, StageOp producer) -> bool;

  /// Erases @p consumer from the cache.
  ///
  /// @return   Whether the dependencies changed as a result.
  auto erase(StageOp consumer) -> bool;
  /// Erases the dependency of @p consumer on @p producer from the cache.
  ///
  /// @return   Whether the dependencies changed as a result.
  auto erase(StageOp consumer, StageOp producer) -> bool;

 private:
  DenseMap<StageOp, SmallPtrSet<StageOp, 4>> cache_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEDEPENDENCY_H_
