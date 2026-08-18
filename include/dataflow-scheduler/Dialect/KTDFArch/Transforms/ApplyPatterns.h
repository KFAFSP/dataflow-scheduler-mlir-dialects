//===-- ApplyPatterns.h -----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_APPLYPATTERNS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_APPLYPATTERNS_H_

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

class PDLPatternModule;

}  // namespace mlir

namespace mlir::ktdf_arch {

class Device;

/// Registers the native PDL functions used by `ktdf_arch` in @p patterns .
void registerNativeFunctions(PDLPatternModule& patterns);

/// Collects the @p patterns matching @p enabled_groups in @p device .
///
/// This function also installs the necessary native constraint and rewrite
/// handlers into @p patterns via `registerNativeFunctions`.
///
/// @return Number of patterns added to @p patterns .
auto getPatterns(const Device& device, PDLPatternModule& patterns,
                 const StringSet<>& enabled_groups) -> size_t;
/// @copydoc getPatterns(const Device&, PDLPatternModule &, const StringSet<>&)
auto getPatterns(const Device& device, PDLPatternModule& patterns,
                 ArrayRef<StringRef> enabled_groups = {}) -> size_t;

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_APPLYPATTERNS_H_
