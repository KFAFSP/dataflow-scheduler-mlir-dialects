//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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
// This file declares all the ktdf_arch dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ApplyPatternsPass : Pass<"ktdfarch-apply-patterns", "func::FuncOp"> {
  let summary = "Applies patterns defined in a `ktdf_arch.device`.";
  let description = [{
    Given a root operation that is mapped to a `ktdf_arch.device`, this pass
    applies all the (enabled groups of) patterns defined within the device to
    the IR nested below the root operation.
  }];

  let options = [
    ListOption<"enabled_groups", "groups", "std::string", 
      "List of pattern groups to enable, or empty for ungrouped patterns only">
  ];

  let dependentDialects = [
    "mlir::ktdf_arch::KTDFArchDialect", 
    "mlir::func::FuncDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_PASSES_TD
