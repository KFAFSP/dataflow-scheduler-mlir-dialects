//===-- ktdf-archgen.cpp ----------------------------------------*- c++ -*-===//
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
// This file implements a stand-alone tool for checking and baking `ktdf_arch`
// dialect architecture specifications.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>
#include <variant>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

llvm::cl::OptionCategory tool_options("Tool Options");
llvm::cl::OptionCategory debug_options("Debug Options");

llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"),
                                          llvm::cl::cat(tool_options));

llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename, omit for checking only"),
    llvm::cl::value_desc("filename"), llvm::cl::cat(tool_options));

llvm::cl::opt<bool> emit_bytecode(
    "emit-bytecode", llvm::cl::desc("Emit bytecode when generating output"),
    llvm::cl::init(false), llvm::cl::cat(tool_options));

llvm::cl::opt<SourceMgrDiagnosticVerifierHandler::Level> verify_diagnostics(
    "verify-diagnostics", llvm::cl::ValueOptional,
    llvm::cl::desc(
        "Check that emitted diagnostics match expected-* lines on the "
        "corresponding line"),
    llvm::cl::values(
        clEnumValN(SourceMgrDiagnosticVerifierHandler::Level::All, "all",
                   "Check all diagnostics (expected, unexpected, near-misses)"),
        clEnumValN(SourceMgrDiagnosticVerifierHandler::Level::All, "",
                   "Check all diagnostics (expected, unexpected, near-misses)"),
        clEnumValN(SourceMgrDiagnosticVerifierHandler::Level::OnlyExpected,
                   "only-expected", "Check only expected diagnostics")),
    llvm::cl::cat(debug_options), llvm::cl::Hidden);

/// Marks @p opt as a hidden debug option.
void makeDebugOption(llvm::cl::Option* opt) {
  if (!opt) {
    return;
  }

  opt->addCategory(debug_options);
  opt->setHiddenFlag(llvm::cl::OptionHidden::Hidden);
}

/// Returns `true` when the tool should only verify the generated diagnostics.
[[nodiscard]] auto shouldVerifyDiagnostics() -> bool {
  return verify_diagnostics.getNumOccurrences() > 0;
}

/// Returns `true` when the tool should not produce any output.
[[nodiscard]] auto isOutputDisabled() -> bool {
  return output_filename.getNumOccurrences() == 0 || shouldVerifyDiagnostics();
}

using DiagHandler = std::variant<SourceMgrDiagnosticHandler,
                                 SourceMgrDiagnosticVerifierHandler>;

[[nodiscard]] auto createDiagHandler(llvm::SourceMgr& source_mgr,
                                     MLIRContext& context) -> DiagHandler {
  if (shouldVerifyDiagnostics()) {
    context.printOpOnDiagnostic(false);
    return DiagHandler(std::in_place_type<SourceMgrDiagnosticVerifierHandler>,
                       source_mgr, &context, verify_diagnostics);
  }

  context.printOpOnDiagnostic(true);
  return DiagHandler(std::in_place_type<SourceMgrDiagnosticHandler>, source_mgr,
                     &context);
}

auto verifyDiagnosticsOrFail(DiagHandler& diag_handler) -> LogicalResult {
  if (auto* const handler =
          std::get_if<SourceMgrDiagnosticVerifierHandler>(&diag_handler);
      handler) {
    return handler->verify();
  }
  return failure();
}

auto check(ModuleOp module) -> LogicalResult {
  // TODO: Implement complex verification here. Presently, only the verifier
  //       runs. On failure, this function is not reached.
  std::ignore = module;
  return success();
}

auto bake(OwningOpRef<ModuleOp> module) -> FailureOr<OwningOpRef<ModuleOp>> {
  PassManager passes(module->getOperation()->getName(),
                     PassManager::Nesting::Implicit);

  passes.addPass(createInstantiateNeighborhoodsPass());
  passes.addPass(createCanonicalizerPass());

  if (failed(passes.run(*module))) {
    return failure();
  }

  return std::move(module);
}

auto runOnInput(MLIRContext& context) -> LogicalResult {
  // Display a warning when interactive input is used.
  if (input_filename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin))) {
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";
  }

  // Open the input file, which is always required.
  std::string error_message;
  auto file = openInputFile(input_filename, &error_message);
  if (!file) {
    return emitError(UnknownLoc::get(&context)) << error_message;
  }

  // Create and set up the llvm::SourceMgr with a diagnostic handler.
  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc{});
  auto diag_handler = createDiagHandler(*source_mgr, context);

  // Parse and verify the input (text or bytecode are both accepted).
  ParserConfig parser_config(&context, true);
  auto module = parseSourceFile<ModuleOp>(source_mgr, parser_config);
  if (!module || failed(check(*module))) {
    // Verification or checking has failed. If this is a --verify-diagnostics
    // run, we may still succeed here if all of them were expected.
    return verifyDiagnosticsOrFail(diag_handler);
  }
  if (isOutputDisabled()) {
    // No output should be produced.
    return success();
  }

  // Open the output file.
  auto output = openOutputFile(output_filename, &error_message);
  if (!output) {
    return emitError(UnknownLoc::get(&context)) << error_message;
  }

  // Run the baking pipeline.
  auto result = bake(std::move(module));
  if (failed(result)) {
    return failure();
  }

  // Write the baked result to the output file.
  if (emit_bytecode) {
    if (output_filename == "-" &&
        llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdout))) {
      llvm::errs() << "warning: printing bytecode to stdout\n";
    }

    BytecodeWriterConfig bytecode_config("ktdf-archgen");
    if (failed(writeBytecodeToFile(result->get(), output->os(),
                                   bytecode_config))) {
      return failure();
    }
  } else {
    AsmState state(result->get());
    result->get().print(output->os(), state);
  }

  // Only keep the output file on disk on success.
  output->keep();
  return success();
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  // Handle command-line arguments.
  llvm::cl::HideUnrelatedOptions({&tool_options, &debug_options});
  makeDebugOption(llvm::cl::getRegisteredOptions()["debug"]);
  makeDebugOption(llvm::cl::getRegisteredOptions()["debug-only"]);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "ktdf-arch architecture generator\n\n"
      "This program runs a checking pipeline on the given input MLIR file "
      "(text or bytecode) and, if -o is given, produces a baked architecture "
      "specification from it.\n");

  // Set up an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect, pdl::PDLDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Invoke the tool.
  return asMainReturnCode(runOnInput(context));
}
