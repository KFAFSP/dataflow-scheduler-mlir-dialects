//===-- Utils.cpp ------------------------------------------------*- c++ -*-==//
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
// This file implements the dataflow dialect utilities.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::dataflow;

auto mlir::dataflow::isIntOrFloatType(Type type) -> bool {
  return isa<CustomMXFloatType, CustomMXIntType, IntegerType, FloatType>(type);
}

auto mlir::dataflow::getIntOrFloatBitWidth(Type type) -> unsigned {
  return llvm::TypeSwitch<Type, unsigned>(type)
      .Case([](CustomMXFloatType custom) { return custom.getWidth(); })
      .Case([](CustomMXIntType custom) { return custom.getWidth(); })
      .Default([](Type type) { return type.getIntOrFloatBitWidth(); });
}

auto mlir::dataflow::getNumElements(Type type) -> int64_t {
  return llvm::TypeSwitch<Type, std::size_t>(type)
      .Case([](VectorType vector) { return vector.getNumElements(); })
      .Case([](MemRefType memref) { return memref.getNumElements(); })
      .Case([](CustomVectorType custom) { return custom.getNumElements(); });
}

int mlir::dataflow::getCoreletId(const GetUnitOp op) {
  int corelet = -1;
  if (!op) return corelet;
  auto corelet_attr = op->getAttrOfType<IntegerAttr>("corelet");
  if (corelet_attr) {
    corelet = corelet_attr.getInt();
  }
  return corelet;
}
