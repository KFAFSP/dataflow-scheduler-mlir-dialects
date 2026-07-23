//===-- Utils.h -------------------------------------------------*- c++ -*-===//
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

#include <doctest/doctest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

struct DiagRecorder : ScopedDiagnosticHandler {
  explicit DiagRecorder(MLIRContext* context, bool fail_on_error)
      : ScopedDiagnosticHandler(context) {
    setHandler([=](Diagnostic& diag) -> LogicalResult {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << diag;
      INFO(message);
      if (fail_on_error && diag.getSeverity() == DiagnosticSeverity::Error) {
        FAIL_CHECK("MLIR error diagnostic encountered");
      }

      return failure();
    });
  }
};

inline auto parse(MLIRContext* context, StringRef filename)
    -> OwningOpRef<ModuleOp> {
  DiagRecorder recorder(context, true);
  ParserConfig config{context, true};
  if (filename.empty()) {
    return parseSourceString<ModuleOp>("", config);
  }

  auto module = parseSourceFile<ModuleOp>(filename, config);
  REQUIRE(module);
  return module;
}

template <class T>
[[nodiscard]] inline auto unorderedEquals(SmallVector<T> lhs,
                                          SmallVector<T> rhs) -> bool {
  llvm::sort(lhs);
  llvm::sort(rhs);
  return lhs == rhs;
}
