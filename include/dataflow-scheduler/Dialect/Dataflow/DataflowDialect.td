//===-- DataflowDialect.td ---------------------------------*- tablegen -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD

include "mlir/IR/OpBase.td"

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::mlir::dataflow";
  
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 0;

  let dependentDialects = [
    "arith::ArithDialect",
    // These are used by Utils.cpp, and may require loading.
    "affine::AffineDialect",
    "scf::SCFDialect"
  ];

  let extraClassDeclaration = [{
    virtual auto parseType(DialectAsmParser &parser) const -> Type override;
    virtual void printType(Type type, DialectAsmPrinter &printer) const override;

    void registerOps();
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD
