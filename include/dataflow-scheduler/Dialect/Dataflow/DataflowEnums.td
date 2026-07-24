//===-- DataflowEnums.td ------------------------------------*- tablegen -*-==//
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
// This file defines the enumerations in the dataflow dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWENUMS_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWENUMS_TD

include "mlir/IR/EnumAttr.td"

def DataflowRoutingDirPseudoRandom : I32EnumAttrCase<"PseudoRandom", 0>;
def DataflowRoutingDirCounterClockwise : I32EnumAttrCase<"CounterClockwise", 1>;
def DataflowRoutingDirClockwise : I32EnumAttrCase<"Clockwise", 2>;
def DataflowRoutingDirBoth : I32EnumAttrCase<"BothWays", 3>;

// FIXME: Why is the C++ enum not called RoutingDirection?

def DataflowRoutingDirection : I32EnumAttr<"DataflowRoutingDirection",
    "dataflow routing direction:",
    [
      DataflowRoutingDirPseudoRandom,
      DataflowRoutingDirCounterClockwise,
      DataflowRoutingDirClockwise,
      DataflowRoutingDirBoth
    ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::dataflow";
}

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWENUMS_TD
