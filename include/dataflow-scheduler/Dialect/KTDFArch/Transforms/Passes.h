//===-- Passes.h ------------------------------------------------*- c++ -*-===//
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
// This file declares all ktdf_arch dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_H_

#include <mlir/Pass/Pass.h>

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace mlir::ktdf_arch {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h.inc"

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_H_
