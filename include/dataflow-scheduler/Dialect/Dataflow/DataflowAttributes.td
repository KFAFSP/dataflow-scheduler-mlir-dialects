//===-- DataflowAttributes.td ------------------------------*- tablegen -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWATTRIBUTES_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWATTRIBUTES_TD

include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.td"
include "dataflow-scheduler/Dialect/Dataflow/DataflowEnums.td"

def DataflowRoutingDirectionAttr : EnumAttr<Dataflow_Dialect, DataflowRoutingDirection, "direction">;

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWATTRIBUTES_TD
