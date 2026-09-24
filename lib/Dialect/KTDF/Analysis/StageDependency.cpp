//===-- StageDependency.cpp -------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageDependency.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

void StageDependency::getProducers(StageOp consumer,
                                   SmallPtrSetImpl<StageOp>& producers,
                                   bool transitive) {
  if (!transitive) {
    for (auto in : consumer.getDependsIn()) {
      for (auto& use : in.getUses()) {
        if (auto producer = dyn_cast<StageOp>(use.getOwner());
            producer && producer.isOutDependency(use)) {
          producers.insert(producer);
        }
      }
    }
    return;
  }

  SmallVector<StageOp> work_list{consumer};
  while (!work_list.empty()) {
    for (auto in : work_list.pop_back_val().getDependsIn()) {
      for (auto& use : in.getUses()) {
        if (auto producer = dyn_cast<StageOp>(use.getOwner());
            producer && producer.isOutDependency(use)) {
          if (producers.insert(producer).second) {
            work_list.push_back(producer);
          }
        }
      }
    }
  }
}

void StageDependency::getConsumers(StageOp producer,
                                   SmallPtrSetImpl<StageOp>& consumers,
                                   bool transitive) {
  if (!transitive) {
    for (auto out : producer.getDependsOut()) {
      for (auto& use : out.getUses()) {
        if (auto consumer = dyn_cast<StageOp>(use.getOwner());
            consumer && consumer.isInDependency(use)) {
          consumers.insert(consumer);
        }
      }
    }
    return;
  }

  SmallVector<StageOp> work_list{producer};
  while (!work_list.empty()) {
    for (auto out : work_list.pop_back_val().getDependsOut()) {
      for (auto& use : out.getUses()) {
        if (auto consumer = dyn_cast<StageOp>(use.getOwner());
            consumer && consumer.isInDependency(use)) {
          if (consumers.insert(consumer).second) {
            work_list.push_back(consumer);
          }
        }
      }
    }
  }
}

auto StageDependency::contains(StageOp consumer, StageOp producer,
                               bool transitive) -> bool {
  const auto get = [&](StageOp stage) -> const SmallPtrSetImpl<StageOp>& {
    const auto [it, invalid] = cache_.try_emplace(consumer);
    if (invalid) {
      getProducers(consumer, it->second);
    }

    return it->second;
  };

  if (!transitive) {
    return get(consumer).contains(producer);
  }

  SmallVector<StageOp> work_list{consumer};
  SmallPtrSet<StageOp, 16> seen{consumer};
  while (!work_list.empty()) {
    const auto consumer = work_list.pop_back_val();
    for (auto transitive : get(consumer)) {
      if (transitive == producer) {
        return true;
      }
      if (seen.insert(transitive).second) {
        work_list.push_back(transitive);
      }
    }
  }

  return false;
}

auto StageDependency::insert(StageOp consumer, StageOp producer) -> bool {
  const auto [it, invalid] = cache_.try_emplace(consumer);
  if (invalid) {
    getProducers(consumer, it->second);
  }

  return it->second.insert(producer).second;
}

auto StageDependency::erase(StageOp consumer) -> bool {
  auto changed = cache_.erase(consumer);
  for (auto&& [other, producers] : cache_) {
    changed |= producers.erase(consumer);
  }
  return changed;
}

auto StageDependency::erase(StageOp consumer, StageOp producer) -> bool {
  if (const auto it = cache_.find(consumer); it != cache_.end()) {
    return it->second.erase(producer);
  }
  return false;
}
