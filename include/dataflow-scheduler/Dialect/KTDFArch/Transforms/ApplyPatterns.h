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
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Mutex.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>

#include <initializer_list>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

namespace mlir::ktdf_arch {

/// Registers the native PDL functions used by `ktdf_arch` in @p patterns .
void registerNativeFunctions(PDLPatternModule& patterns);

/// Key type that selects groups of patterns.
class PatternGroups {
  using impl_type = SmallVector<StringRef>;

 public:
  /*implicit*/ PatternGroups(ArrayRef<StringRef> groups)
      : PatternGroups(llvm::from_range, groups) {}
  /*implicit*/ PatternGroups(std::initializer_list<StringRef> groups)
      : PatternGroups(llvm::from_range, groups) {}
  template <class Range>
  explicit PatternGroups(llvm::from_range_t, Range&& range)
      : PatternGroups(llvm::adl_begin(range), llvm::adl_end(range)) {}
  template <class InputIt>
  explicit PatternGroups(InputIt begin, InputIt end) {
    SmallVector<std::string> groups(begin, end);
    initialize(groups);
  }

  [[nodiscard]] auto contains(StringRef group) const -> bool;

  [[nodiscard]] auto asStringRef() const -> StringRef { return key_; }
  /*implict*/ operator StringRef() const { return asStringRef(); }

  [[nodiscard]] auto operator==(const PatternGroups& rhs) const -> bool {
    return key_ == rhs.key_;
  }
  [[nodiscard]] auto operator!=(const PatternGroups& rhs) const -> bool {
    return !(*this == rhs);
  }
  [[nodiscard]] auto operator<(const PatternGroups& rhs) const -> bool {
    return key_ < rhs.key_;
  }

  [[nodiscard]] friend auto hash_code(const PatternGroups& self)
      -> llvm::hash_code {
    return llvm::hash_value(self.key_);
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = impl_type::value_type;
  using size_type = impl_type::size_type;
  using iterator = impl_type::const_iterator;

  [[nodiscard]] auto empty() const -> bool { return groups_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return groups_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return groups_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return groups_.end(); }

  [[nodiscard]] auto asArrayRef() const -> ArrayRef<StringRef> {
    return groups_;
  }
  /*implict*/ operator ArrayRef<StringRef>() const { return asArrayRef(); }

 private:
  void initialize(SmallVectorImpl<std::string>& groups);

  SmallString<32> key_;
  impl_type groups_;
};

/// Collects the @p patterns matching @p enabled_groups in @p device .
///
/// This function also installs the necessary native constraint and rewrite
/// handlers into @p patterns via `registerNativeFunctions`.
///
/// @return Number of patterns added to @p patterns .
auto getPatterns(const Device& device, PDLPatternModule& patterns,
                 const PatternGroups& enabled_groups) -> size_t;

/// Caches assembled groups of rewrite `ktdf_arch.patterns`.
class PatternCache : public DeviceView {
  using map_type = std::map<PatternGroups, FrozenRewritePatternSet>;

 public:
  explicit PatternCache(const Device& device) : DeviceView(device) {}

  /// Gets the patterns in @p enabled_groups .
  [[nodiscard]] auto get(const PatternGroups& enabled_groups)
      -> FrozenRewritePatternSet;

 private:
  llvm::sys::SmartMutex<true> mutex_;
  map_type map_;
};

/// Creates a `ktdfarch-apply-patterns` pass instance for @p enabled_groups .
[[nodiscard]] auto createApplyPatternsPass(
    std::initializer_list<StringRef> enabled_groups) -> std::unique_ptr<Pass>;

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_TRANSFORMS_APPLYPATTERNS_H_
