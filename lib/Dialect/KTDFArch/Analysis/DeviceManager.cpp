//===-- DeviceManager.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Mutex.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/FileUtilities.h>

#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

auto toStringView(StringRef ref) -> std::string_view {
  return std::string_view(ref.data(), ref.size());
}

auto resolveImportPath(DeviceOp importer) -> std::filesystem::path {
  std::filesystem::path import_path(toStringView(*importer.getImportPath()));
  if (import_path.is_absolute()) {
    return import_path;
  }

  auto module = importer->getParentOfType<ModuleOp>();
  assert(module);
  const auto loc = dyn_cast_if_present<FileLineColLoc>(module->getLoc());
  if (!loc || loc.getFilename() == "-") {
    // Path is relative to the CWD.
    return import_path;
  }

  std::filesystem::path importer_path(toStringView(loc.getFilename()));
  return importer_path.replace_filename(import_path);
}

auto parseDevice(MLIRContext* context,
                 std::unique_ptr<llvm::MemoryBuffer> buffer,
                 StringAttr device_name) -> OwningOpRef<DeviceOp> {
  // Create a temporary SourceMgr and attach a diagnostic handler that uses it,
  // so that verification errors will be emitted correctly.
  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(buffer), {});
  SourceMgrDiagnosticHandler child_handler(*source_mgr, context);

  // Parse the input file into a ModuleOp.
  ParserConfig parser_config{context, true};
  auto parsed = parseSourceFile<ModuleOp>(source_mgr, parser_config).release();
  if (!parsed) {
    return {};
  }

  for (auto device : parsed.getOps<DeviceOp>()) {
    if (device.getSymNameAttr() != device_name) {
      continue;
    }
    if (device.isImported()) {
      device.emitOpError("recursive importing is prohibited");
      return {};
    }

    // Remove the DeviceOp from the parent, which will be deleted when the
    // scope exits. The caller will take care of it.
    device->remove();
    return OwningOpRef<DeviceOp>(device);
  }

  return {};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

Device::Device(DeviceOp declaration)
    : DeviceOp(nullptr), declaration_(declaration) {
  auto& definition = static_cast<DeviceOp&>(*this);

  if (!declaration.isImported()) {
    definition = declaration;
    return;
  }

  // The graph still has to be imported. Open the file, parse it, and emit an
  // error if it doesn't work.
  const auto import_path = resolveImportPath(declaration);
  std::string error_message;
  auto maybe_file = openInputFile(import_path.native(), &error_message);
  if (!maybe_file) {
    declaration.emitOpError("unable to import device graph: ") << error_message;
    return;
  }
  auto maybe_definition =
      parseDevice(declaration->getContext(), std::move(maybe_file),
                  declaration.getSymNameAttr());
  if (!maybe_definition) {
    declaration.emitOpError("unable to import device graph: no device named '")
        << declaration.getName() << "'";
    return;
  }

  // Overwrite discardable attributes using the importing DeviceOp.
  NamedAttrList discardable_attrs(
      maybe_definition.get()->getDiscardableAttrDictionary());
  for (const auto& attr : declaration->getDiscardableAttrDictionary()) {
    discardable_attrs.set(attr.getName(), attr.getValue());
  }
  maybe_definition.get()->setDiscardableAttrs(discardable_attrs);

  definition = maybe_definition.release();
}

Device::~Device() {
  auto definition = getDefinition();
  if (definition && definition != getDeclaration()) {
    definition->erase();
  }
}

auto Device::getLoc() const -> Location {
  const auto decl_loc = getDeclaration()->getLoc();
  if (auto definition = getDefinition();
      definition && definition != getDeclaration()) {
    FusedLoc::get({decl_loc, definition->getLoc()},
                  StringAttr::get(getContext(), "imported"), getContext());
  }

  return decl_loc;
}

//===----------------------------------------------------------------------===//
// DeviceView
//===----------------------------------------------------------------------===//

DeviceView::~DeviceView() = default;

//===----------------------------------------------------------------------===//
// DeviceManager
//===----------------------------------------------------------------------===//

auto DeviceManager::getOrImportDevice() -> const Device* {
  {
    // Return the only imported device, if any.
    const auto it = devices_.begin();
    if (it != devices_.end()) {
      if (std::next(it) != devices_.end()) {
        return nullptr;
      }

      return &it->second;
    }
  }

  // Find exactly one declaration in the root.
  if (root_->getNumRegions() != 1) {
    return nullptr;
  }
  const auto decls = root_->getRegion(0).getOps<DeviceOp>();
  if (decls.empty() || std::next(decls.begin()) != decls.end()) {
    return nullptr;
  }

  // Import that declaration, which will put it into the map to be found again.
  return &getOrImportDevice(*decls.begin());
}

auto DeviceManager::getOrImportDevice(StringAttr name) -> const Device* {
  // Try to return the imported device.
  const auto it = devices_.find(name);
  if (it != devices_.end()) {
    return &it->second;
  }

  // Try to import the matching declaration.
  if (root_->getNumRegions() != 1) {
    return nullptr;
  }
  for (auto decl : root_->getRegion(0).getOps<DeviceOp>()) {
    if (decl.getNameAttr() == name) {
      return &getOrImportDevice(decl);
    }
  }

  return nullptr;
}

auto DeviceManager::getOrImportDevice(DeviceOp declaration) -> const Device& {
  llvm::sys::SmartScopedLock<true> lock(mutex_);

  return devices_.try_emplace(declaration.getNameAttr(), declaration)
      .first->second;
}

auto DeviceManager::getOrCreateViewImpl(
    const Device& device, TypeID type_id,
    function_ref<std::unique_ptr<DeviceView>(const Device&)> ctor)
    -> DeviceView& {
  llvm::sys::SmartScopedLock<true> lock(mutex_);

  const auto key = std::make_pair(&device, type_id);
  auto [it, invalid] = views_.try_emplace(key, nullptr);
  if (invalid) {
    it->second = ctor(device);
  }

  return *it->second;
}

//===----------------------------------------------------------------------===//
// DeviceManagerRef
//===----------------------------------------------------------------------===//

DeviceManagerRef::DeviceManagerRef(Operation* root)
    : mlir::ktdf_arch::DeviceManagerRef(new DeviceManager(root), true) {}

DeviceManagerRef::DeviceManagerRef(Operation* root, AnalysisManager analyses) {
  if (const auto maybe_manager =
          analyses.getCachedParentAnalysis<DeviceManager>(root);
      maybe_manager) {
    ptr_.setPointerAndInt(&maybe_manager->get(), 0);
    return;
  }

  ptr_.setPointerAndInt(new DeviceManager(root), 1);
}

DeviceManagerRef::~DeviceManagerRef() {
  if (ptr_.getInt() != 0) {
    delete ptr_.getPointer();
  }
}

//===----------------------------------------------------------------------===//
// DeviceRef
//===----------------------------------------------------------------------===//

DeviceRef::DeviceRef(DeviceOp declaration, AnalysisManager analyses)
    : manager_(declaration->getParentOfType<ModuleOp>(), analyses),
      device_(manager_->getOrImportDevice(declaration)) {}

//===----------------------------------------------------------------------===//
// ktdf_arch::findDeviceDeclarationFor
//===----------------------------------------------------------------------===//

auto ktdf_arch::findDeviceDeclarationFor(Operation* op) -> DeviceOp {
  DeviceOp only = nullptr;

  // Walk upwards until we find an unambiguous mapping.
  while (op) {
    // Resolve a ktdf_arch.maps_to attribute that references a device.
    if (const auto maps_to =
            dyn_cast_if_present<SymbolRefAttr>(getProperty<MapsToAttr>(op));
        maps_to) {
      if (const auto device =
              SymbolTable::lookupNearestSymbolFrom<DeviceOp>(op, maps_to);
          device) {
        return device;
      }
    }

    if (auto module = dyn_cast<ModuleOp>(op); module) {
      const auto devices = module.getOps<DeviceOp>();
      if (!devices.empty()) {
        if (!only && std::next(devices.begin()) == devices.end()) {
          // The module contains exactly one device declaration, it could be
          // the only reachable one.
          only = *devices.begin();
        }

        // There are at least 2 reachable device declarations.
        return nullptr;
      }
    }

    op = op->getParentOp();
  }

  // If we found only one reachable device declaration, it must be that one.
  return only;
}
