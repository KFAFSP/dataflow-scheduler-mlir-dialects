//===----------------------------------------------------------------------===//
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

#include "UseDeviceManagerPass.h"

#include <mlir/IR/BuiltinOps.h>

#include <memory>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

struct UseDeviceManagerPass : OperationPass<> {
  static constexpr auto kAddrAttrName = "device_manager.addr";

  UseDeviceManagerPass()
      : OperationPass<>(TypeID::get<UseDeviceManagerPass>()) {}

  auto getName() const -> StringRef override { return "UseDeviceManagerPass"; }

  auto clonePass() const -> std::unique_ptr<Pass> override {
    return std::make_unique<UseDeviceManagerPass>(*this);
  }

  void runOnOperation() override {
    if (auto module = dyn_cast<ModuleOp>(getOperation()); module) {
      auto& devices = getAnalysis<DeviceManager>();

      if (module->hasAttr(kAddrAttrName)) {
        checkAddresses(module, devices);
      } else {
        storeAddresses(module, devices);
      }
      return;
    }

    auto module = getOperation()->getParentOfType<ModuleOp>();

    const auto maybe_devices = getCachedParentAnalysis<DeviceManager>(module);
    if (!maybe_devices) {
      signalPassFailure();
      return;
    }

    checkAddresses(module, maybe_devices->get());
  }

  void storeAddresses(ModuleOp module, DeviceManager& devices) {
    Builder builder(getOperation());

    llvm::SmallVector<NamedAttribute> device_addrs;
    for (auto declaration : module.getOps<DeviceOp>()) {
      const auto& device = devices.getOrImportDevice(declaration);
      declaration->setAttr(
          kAddrAttrName,
          builder.getI64IntegerAttr(reinterpret_cast<std::uintptr_t>(&device)));
    }
  }

  void checkAddresses(ModuleOp module, DeviceManager& devices) {
    for (auto declaration : module.getOps<DeviceOp>()) {
      checkAddress(declaration, devices);
    }
  }

  void checkAddress(DeviceOp declaration, DeviceManager& devices) {
    const auto& device = devices.getOrImportDevice(declaration);
    if (reinterpret_cast<std::uintptr_t>(&device) !=
        declaration->getAttrOfType<IntegerAttr>(kAddrAttrName)
            .getValue()
            .getZExtValue()) {
      signalPassFailure();
      return;
    }
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseDeviceManagerPass);
};

}  // namespace

auto mlir::ktdf_arch::test::createUseDeviceManagerPass()
    -> std::unique_ptr<Pass> {
  return std::make_unique<UseDeviceManagerPass>();
}
