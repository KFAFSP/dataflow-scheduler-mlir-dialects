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

#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/AnalysisManager.h>

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
  /// Gets or imports the Device for @p declaration .
  ///
  /// If any error occurs, a diagnostic is raised at the declaration. If an
  /// error occurs parsing the imported file, a temporary diagnostic handler
  /// will emit it based on the imported file. On any error, the resulting
  /// Device will be contextually convertible to `false`.
  explicit Device(DeviceOp declaration);

  // Devices are neither movable nor copyable.
  Device(Device&& move) = delete;
  auto operator=(Device&& move) -> Device& = delete;
  Device(const Device&) = delete;
  auto operator=(const Device&) -> Device& = delete;

  // Destroys the Device, freeing its imported definition (if any).
  ~Device();

  static auto isInvalidated(const AnalysisManager::PreservedAnalyses& /*pa*/)
      -> bool {
    // Devices within a module and their import locations must remain immutable,
    // and therefore the cached instances are always preserved.
    return false;
  }

  using DeviceOp::getBody;
  using DeviceOp::getBodyRegion;
  using DeviceOp::getName;
  using DeviceOp::getNameAttr;
  using DeviceOp::getVersion;

  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return getDeclaration()->getContext();
  }

  [[nodiscard]] auto getAttr(StringRef name) const -> Attribute {
    return getDefinition()->getAttr(name);
  }
  template <class Attr>
  [[nodiscard]] auto getAttrOfType(StringRef name) const -> Attr {
    return getDefinition()->getAttrOfType<Attr>(name);
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
  [[nodiscard]] auto operator*() -> DeviceOp { return declaration_; }

  friend void swap(Device& lhs, Device& rhs) {
    using std::swap;
    swap(static_cast<DeviceOp&>(lhs), static_cast<DeviceOp&>(rhs));
    swap(lhs.declaration_, rhs.declaration_);
  }

 private:
  explicit Device(DeviceOp declaration, DeviceOp definition);

  DeviceOp declaration_;
};

/// Manages cached access to defined or imported devices.
class DeviceManager {
 public:
  /// Creates a DeviceManager for all devices nested directly below @p op .
  explicit DeviceManager(Operation* op, AnalysisManager& analyses);

  // DeviceManagers are neither movable nor copyable.
  DeviceManager(const DeviceManager&) = delete;
  auto operator=(const DeviceManager&) -> DeviceManager& = delete;
  DeviceManager(DeviceManager&&) = delete;
  auto operator=(DeviceManager&&) -> DeviceManager& = delete;

  /// Destroys the DeviceManager and all its imported devices.
  ~DeviceManager() = default;

  /// Gets or imports the only Device in the current scope.
  ///
  /// See getOrImportDevice(DeviceOp) for more information on importing.
  ///
  /// @retval Device    Only device in the current scope.
  /// @retval nullptr   No or more than one device found, or importing failed.
  auto getOrImportDevice() -> Device*;
  /// Gets the Device for @p name , importing it if necessary.
  ///
  /// See getOrImportDevice(DeviceOp) for more information on importing.
  ///
  /// @retval Device    Device with @p name .
  /// @retval nullptr   No such device found, or importing failed.
  auto getOrImportDevice(StringAttr name) -> Device*;
  /// @copydoc getOrImportDevice(StringAttr)
  auto getOrImportDevice(StringRef name) -> Device* {
    return getOrImportDevice(StringAttr::get(root_->getContext(), name));
  }
  /// Gets or imports the Device referenced by @p declaration .
  ///
  /// The first time an imported device is queried via the DeviceManager, it is
  /// imported from the file, linked to the manager's lifetime and cached.
  ///
  /// See Device::getOrImport(DeviceOp) for more information on importing.
  ///
  /// @retval Decive    Device for @p declaration .
  /// @retval nullptr   Importing failed or invalid @p declaration .
  auto getOrImportDevice(DeviceOp declaration) -> Device*;

 private:
  auto getOrImportDeviceImpl(DeviceOp declaration) -> Device*;

  Operation* root_;
  AnalysisManager analyses_;
  DenseMap<StringAttr, DeviceOp> declarations_;
};

/// Base class for implementing views over Devices.
class DeviceView {
 public:
  explicit DeviceView(DeviceOp declaration, AnalysisManager& analyses);

  static auto isInvalidated(const AnalysisManager::PreservedAnalyses& /*pa*/)
      -> bool {
    // By default, devices are considered immutable. Clients that change the
    // device IR must update the views themselves.
    return false;
  }

  [[nodiscard]] auto getContext() const -> MLIRContext* {
    return device_.getDeclaration()->getContext();
  }
  [[nodiscard]] auto getDevice() const -> Device& { return device_; }

 private:
  Device& device_;
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_ANALYSIS_DEVICEMANAGER_H_