//===-- Symbol.h -------------------------------------------------*- c++ -*-==//
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
// This file includes the entire symbol dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_H_
#define DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_H_

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/OpTraits.h" // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// Tablegen Declarations
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Symbol/SymbolDialect.h.inc"
#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h.inc"

#endif  // DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_H_
