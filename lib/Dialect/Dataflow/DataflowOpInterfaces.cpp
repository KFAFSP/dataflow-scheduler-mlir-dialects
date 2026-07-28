//===-- DataflowOpInterfaces.cpp --------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/Dataflow/DataflowInterfaces.h"

using namespace mlir;
using namespace mlir::dataflow;

auto mlir::dataflow::getDbgNameAttr(Operation* op) -> StringAttr {
  if (auto iface = dyn_cast<DebugNameOpInterface>(op); iface) {
    return iface.getDbgNameAttr();
  }

  if (const auto result =
          op->getAttrOfType<StringAttr>(DebugNameOpInterface::kDbgNameAttrName);
      result) {
    return result;
  }

  return nullptr;
}

void mlir::dataflow::setDbgNameAttr(Operation* op, StringAttr name) {
  if (auto iface = dyn_cast<DebugNameOpInterface>(op)) {
    iface.setDbgNameAttr(name);
    return;
  }

  if (name != nullptr) {
    op->setAttr(DebugNameOpInterface::kDbgNameAttrName, name);
    return;
  }

  op->removeAttr(DebugNameOpInterface::kDbgNameAttrName);
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/DataflowOpInterfaces.cpp.inc"
