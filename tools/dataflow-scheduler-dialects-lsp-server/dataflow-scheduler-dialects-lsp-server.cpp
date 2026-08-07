//===-- dataflow-scheduler-dialects-lsp-server.cpp --------------*- c++ -*-===//
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
// This file implements an MLIR LSP server for development purposes.
//
//===----------------------------------------------------------------------===//

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLoweringDialect.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"

using namespace mlir;

auto main(int argc, char** argv) -> int {
  DialectRegistry registry;
  registry.insert<agen::AgenDialect, dataflow::DataflowDialect,
                  ktdf::KTDFDialect, ktdf_arch::KTDFArchDialect,
                  ktdf_lowering::KTDFLoweringDialect, symbol::SymbolDialect,
                  uniform::UniformDialect, vectorchain::VectorChainDialect>();
  registerAllDialects(registry);
  registerAllExtensions(registry);

  return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
