//===-- DataflowAttributes.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// DataflowDialect
//===----------------------------------------------------------------------===//

void DataflowDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.cpp.inc"
      >();
}
