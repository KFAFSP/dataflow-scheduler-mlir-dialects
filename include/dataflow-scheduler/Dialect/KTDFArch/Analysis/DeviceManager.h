//===-- DeviceManager.h -----------------------------------------*- c++ -*-===//
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
// This file declares the DeviceManager analysis.
//
// Every ktdf_arch device used within a module must have a declaration within,
// but ktdf_arch.device ops with an import_path attribute set do not contain
// the actual device definition. This definition must be loaded from the
// indicated (absolute or module-relative) path first. Imports also prevent the
// IR debug output from being inundated with repetitive device definitions.
//
// The DeviceManager is a module-level analysis that can be used by clients to
// automatically handle the importing of devices and manage the lifetime of the
// imported definitions.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_DEVICEMANAGER_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_DEVICEMANAGER_H_

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Mutex.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/TypeID.h>

#include <memory>
#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace mlir::ktdf_arch {

/// Owns a ktdf_arch device definition.
///
/// Clients should not use the DeviceOp returned by getDefinition() when
/// interacting with parts of MLIR that rely on a strongly-connected module,
/// since it might be free-standing when the definition is imported. Instead,
/// clients should always use Device instances, passed by reference.
///
/// Devices can be cached as an analysis on the declaring DeviceOp.
class Device : private DeviceOp {
 public:
  /// Initializes a @c nullptr like device.
  /*implicit*/ Device() = default;
  /// Gets or imports the Device for @p declaration .
  ///
  /// If any error occurs, a diagnostic is raised at the declaration. If an
  /// error occurs parsing the imported file, a temporary diagnostic handler
  /// will emit it based on the imported file. On any error, the resulting
  /// Device will be contextually convertible to `false`.
  explicit Device(DeviceOp declaration);

  /*implicit*/ Device(Device&& move) : Device() { swap(*this, move); }

  /// Devices are not assignable or copyable.
  auto operator=(Device&& move) -> Device& = delete;
  Device(const Device&) = delete;
  auto operator=(const Device&) -> Device& = delete;

  // Destroys the Device, freeing its imported definition (if any).
  ~Device();

  [[nodiscard]] auto getBody() const -> Block* {
    return getDefinition() ? getDefinition().getBody() : nullptr;
  }
  [[nodiscard]] auto getBodyRegion() const -> Region& {
    return getDefinition() ? getDefinition().getBodyRegion()
                           : getDeclaration().getBodyRegion();
  }
  [[nodiscard]] auto getName() const -> StringRef {
    return getDeclaration().getName();
  }
  [[nodiscard]] auto getNameAttr() const -> StringAttr {
    return getDeclaration().getNameAttr();
  }
  [[nodiscard]] auto getVersion() const -> std::optional<int64_t> {
    return getDefinition() ? getDefinition().getVersion()
                           : getDeclaration().getVersion();
  }

  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return getDeclaration()->getContext();
  }

  [[nodiscard]] auto getAttr(StringRef name) const -> Attribute {
    return getDefinition() ? getDefinition()->getAttr(name)
                           : getDeclaration()->getAttr(name);
  }
  template <class Attr>
  [[nodiscard]] auto getAttrOfType(StringRef name) const -> Attr {
    return getDefinition() ? getDefinition()->getAttrOfType<Attr>(name)
                           : getDeclaration()->getAttrOfType<Attr>(name);
  }

  [[nodiscard]] auto isImported() const -> bool {
    return getDeclaration().isImported();
  }
  [[nodiscard]] auto getDeclaration() const -> DeviceOp { return declaration_; }
  [[nodiscard]] auto getDefinition() const -> DeviceOp { return *this; }

  [[nodiscard]] auto operator==(const Device& rhs) const -> bool {
    return declaration_ == rhs.declaration_;
  }
  [[nodiscard]] auto operator!=(const Device& rhs) const -> bool {
    return !(*this == rhs);
  }

  explicit operator bool() const { return static_cast<bool>(getDefinition()); }

  friend void swap(Device& lhs, Device& rhs) {
    using std::swap;
    swap(static_cast<DeviceOp&>(lhs), static_cast<DeviceOp&>(rhs));
    swap(lhs.declaration_, rhs.declaration_);
  }

 private:
  DeviceOp declaration_;
};

/// Base class for implementing views over Devices.
class DeviceView {
 public:
  explicit DeviceView(const Device& device) : device_(device) {};

  virtual ~DeviceView();

  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return device_.getDeclaration()->getContext();
  }
  [[nodiscard]] auto getDevice() const -> const Device& { return device_; }

 private:
  const Device& device_;
};

/// Manages cached access to defined or imported devices.
class DeviceManager {
  using device_map_type = llvm::DenseMap<StringAttr, Device>;
  using view_map_type = llvm::DenseMap<std::pair<const Device*, TypeID>,
                                       std::unique_ptr<DeviceView>>;

 public:
  /// Creates a DeviceManager for all devices nested directly below @p op .
  explicit DeviceManager(Operation* root) : root_(root), mutex_() {}

  // DeviceManagers are neither movable nor copyable.
  DeviceManager(const DeviceManager&) = delete;
  auto operator=(const DeviceManager&) -> DeviceManager& = delete;
  DeviceManager(DeviceManager&&) = delete;
  auto operator=(DeviceManager&&) -> DeviceManager& = delete;

  /// Destroys the DeviceManager and all its imported devices.
  ~DeviceManager() = default;

  static auto isInvalidated(const AnalysisManager::PreservedAnalyses& /*pa*/)
      -> bool {
    // Devices within a module and their import locations must remain immutable,
    // and therefore the cached instances are always preserved.
    return false;
  }

  /// Gets or imports the only Device in the current scope.
  ///
  /// See getOrImportDevice(DeviceOp) for more information on importing.
  ///
  /// @retval Device    Only device in the current scope, testing `false` on
  ///                   import error .
  /// @retval nullptr   No or more than one device found.
  auto getOrImportDevice() -> const Device*;
  /// Gets the Device for @p name , importing it if necessary.
  ///
  /// See getOrImportDevice(DeviceOp) for more information on importing.
  ///
  /// @retval Device    Device with @p name , testing `false` on import error .
  /// @retval nullptr   No such device found, or importing failed.
  auto getOrImportDevice(StringAttr name) -> const Device*;
  /// @copydoc getOrImportDevice(StringAttr)
  auto getOrImportDevice(StringRef name) -> const Device* {
    return getOrImportDevice(StringAttr::get(root_->getContext(), name));
  }
  /// Gets or imports the Device referenced by @p declaration .
  ///
  /// The first time an imported device is queried via the DeviceManager, it is
  /// imported from the file, linked to the manager's lifetime and cached.
  ///
  /// See Device(DeviceOp) for more information on importing.
  ///
  /// @return Device for @p declaration , which tests `false on import error.
  auto getOrImportDevice(DeviceOp declaration) -> const Device&;

  /// Gets or creates the @p View for @p declaration .
  ///
  /// @pre  @p View must be a subclass of `DeviceView`.
  template <class View>
  auto getOrCreateView(DeviceOp declaration) -> View& {
    static_assert(std::is_base_of_v<DeviceView, View>);
    return getOrCreateView<View>(getOrImportDevice(declaration));
  }

  /// Gets or creates the @p View for @p device .
  ///
  /// @pre  @p View must be a subclass of `DeviceView`.
  template <class View>
  auto getOrCreateView(const Device& device) -> View& {
    static_assert(std::is_base_of_v<DeviceView, View>);
    static_assert(std::is_constructible_v<DeviceView, const Device&>);

    return static_cast<View&>(getOrCreateViewImpl(
        device, TypeID::get<View>(),
        [](const Device& device) { return std::make_unique<View>(device); }));
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = Device;
  using size_type = device_map_type::size_type;

  struct iterator
      : llvm::mapped_iterator_base<iterator, device_map_type::const_iterator,
                                   const Device&> {
    using llvm::mapped_iterator_base<iterator, device_map_type::const_iterator,
                                     const Device&>::mapped_iterator_base;

    static auto mapElement(const device_map_type::value_type& pair)
        -> const Device& {
      return pair.second;
    }
  };

  [[nodiscard]] auto empty() const -> bool { return devices_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return devices_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return devices_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return devices_.end(); }

 private:
  auto getOrCreateViewImpl(
      const Device& device, TypeID type_id,
      function_ref<std::unique_ptr<DeviceView>(const Device&)> ctor)
      -> DeviceView&;

  Operation* root_;
  device_map_type devices_;
  view_map_type views_;
  llvm::sys::SmartMutex<true> mutex_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_DEVICEMANAGER_H_