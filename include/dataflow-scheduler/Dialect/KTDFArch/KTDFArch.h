//===-- KTDFArch.h ----------------------------------------------*- c++ -*-===//
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
// This file includes the entire ktdf_lowering dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_H_

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"  // IWYU pragma: export
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"  // IWYU pragma: export
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"  // IWYU pragma: export
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.h"  // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Custom Traits
//===----------------------------------------------------------------------===//

namespace mlir::ktdf_arch {

/// Verifies that @p op is a valid architecture subgraph, i.e., that it only
/// contains other resources (and an optional terminator).
auto verifySubgraph(Operation* op) -> LogicalResult;

/// Trait for operations that contain architecture subgraphs.
template <class Derived>
struct IsSubgraph : mlir::OpTrait::TraitBase<Derived, IsSubgraph> {
  static auto verifyTrait(Operation* op) -> LogicalResult {
    return verifySubgraph(op);
  }
};

/// Trait for operations that provide meta information.
template <class Derived>
struct IsMeta : mlir::OpTrait::TraitBase<Derived, IsMeta> {};

}  // namespace mlir::ktdf_arch

//===----------------------------------------------------------------------===//
// Tablegen Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h.inc"  // IWYU pragma: export

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_H_
