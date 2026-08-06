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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

#include <doctest/doctest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

TEST_CASE("mlir::ktdf_arch::Device") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "device-manager-test.mlir");
  SymbolTable declarations(module.get());

  SUBCASE("get inline_device") {
    auto declaration = declarations.lookup<DeviceOp>("inline_device");
    REQUIRE(declaration);

    DiagRecorder recorder(&context, true);
    Device device(declaration);
    REQUIRE(device);

    CHECK_EQ(device.getDeclaration(), declarations.lookup("inline_device"));
    CHECK_EQ(device.getDeclaration(), device.getDefinition());
  }

  SUBCASE("import missing_device") {
    auto declaration = declarations.lookup<DeviceOp>("missing_device");
    REQUIRE(declaration);

    DiagRecorder recorder(&context, false);
    Device device(declaration);
    REQUIRE_FALSE(device);

    REQUIRE(device.getDeclaration() == declaration);
  }

  SUBCASE("import device") {
    auto declaration = declarations.lookup<DeviceOp>("device");
    REQUIRE(declaration);

    DiagRecorder recorder(&context, true);
    Device device(declaration);
    REQUIRE(device);

    CHECK_EQ(device.getDeclaration(), declarations.lookup("device"));
    CHECK_NE(device.getDeclaration(), device.getDefinition());
    CHECK_EQ(device.getDefinition()->getParentOp(), nullptr);
    CHECK_EQ(device.getVersion(), 1);

    const auto overridable =
        device.getAttrOfType<mlir::IntegerAttr>("overridable");
    REQUIRE(overridable);
    CHECK_EQ(overridable.getValue().getZExtValue(), 2);
  }
}

TEST_CASE("mlir::ktdf_arch::DeviceManager::getOrImportDevice()") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  const auto fixture = [&](StringRef filename, auto&& test) {
    auto module = parse(&context, filename);

    // Setup an AnalysisManager to mock pass execution.
    ModuleAnalysisManager module_analyses(module.get(), nullptr);
    AnalysisManager analysis_manager = module_analyses;
    auto& devices = analysis_manager.getAnalysis<DeviceManager>();

    test(devices);
  };

  SUBCASE("no devices") {
    fixture("", [](DeviceManager& devices) {
      auto* const device = devices.getOrImportDevice();
      REQUIRE_FALSE(device);
    });
  }

  SUBCASE("one device") {
    fixture(INPUTS_DIR "device.mlir", [](DeviceManager& devices) {
      auto* const device = devices.getOrImportDevice();
      REQUIRE(device);
      CHECK(device->getName() == "device");
    });
  }

  SUBCASE("more than one device") {
    fixture(INPUTS_DIR "device-manager-test.mlir", [](DeviceManager& devices) {
      auto* const device = devices.getOrImportDevice();
      REQUIRE_FALSE(device);
    });
  }
}

TEST_CASE("mlir::ktdf_arch::DeviceManager::getOrImportDevice(StringRef)") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "device-manager-test.mlir");
  // Inject a device with an absolute path.
  OpBuilder builder(module->getBodyRegion());
  DeviceOp::create(builder, builder.getUnknownLoc(), "absolute_device",
                   INPUTS_DIR "device-absolute.mlir");
  SymbolTable declarations(module.get());

  // Setup an AnalysisManager to mock pass execution.
  ModuleAnalysisManager module_analyses(module.get(), nullptr);
  AnalysisManager analysis_manager = module_analyses;
  auto& devices = analysis_manager.getAnalysis<DeviceManager>();

  SUBCASE("get inline_device") {
    DiagRecorder recorder(&context, true);
    auto* const device = devices.getOrImportDevice("inline_device");
    REQUIRE(device);

    CHECK_EQ(device->getDeclaration(), declarations.lookup("inline_device"));
    CHECK_EQ(device->getDeclaration(), device->getDefinition());
  }

  SUBCASE("import unknown_device") {
    DiagRecorder recorder(&context, true);
    auto* const device = devices.getOrImportDevice("unknown_device");
    REQUIRE_FALSE(device);
  }

  SUBCASE("import missing_device") {
    DiagRecorder recorder(&context, false);
    auto* const device = devices.getOrImportDevice("missing_device");
    REQUIRE(device);
    CHECK_FALSE(*device);
  }

  SUBCASE("import device") {
    DiagRecorder recorder(&context, true);
    auto* const device = devices.getOrImportDevice("device");
    REQUIRE(device);

    CHECK_EQ(device->getDeclaration(), declarations.lookup("device"));
    CHECK_NE(device->getDeclaration(), device->getDefinition());
    CHECK_EQ(device->getDefinition()->getParentOp(), nullptr);
    CHECK_EQ(device->getVersion(), 1);

    const auto overridable =
        device->getAttrOfType<mlir::IntegerAttr>("overridable");
    REQUIRE(overridable);
    CHECK_EQ(overridable.getValue().getZExtValue(), 2);
  }

  SUBCASE("import absolute_device") {
    DiagRecorder recorder(&context, true);
    auto* const device = devices.getOrImportDevice("absolute_device");
    REQUIRE(device);

    CHECK_EQ(device->getDeclaration(), declarations.lookup("absolute_device"));
    CHECK_NE(device->getDeclaration(), device->getDefinition());
    CHECK_EQ(device->getDefinition()->getParentOp(), nullptr);
  }

  SUBCASE("import invalid_device") {
    DiagRecorder recorder(&context, false);
    auto* const device = devices.getOrImportDevice("invalid_device");
    REQUIRE(device);
    CHECK_FALSE(*device);
  }

  SUBCASE("import recursive_device") {
    DiagRecorder recorder(&context, false);
    auto* const device = devices.getOrImportDevice("recursive_device");
    REQUIRE(device);
    CHECK_FALSE(*device);
  }
}

TEST_CASE("mlir::ktdf_arch::DeviceManager::getOrImportDevice(DeviceOp)") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "device-manager-test.mlir");
  SymbolTable declarations(module.get());

  // Setup an AnalysisManager to mock pass execution.
  ModuleAnalysisManager module_analyses(module.get(), nullptr);
  AnalysisManager analysis_manager = module_analyses;
  auto& devices = analysis_manager.getAnalysis<DeviceManager>();

  SUBCASE("get inline_device") {
    const auto declaration = declarations.lookup<DeviceOp>("inline_device");
    DiagRecorder recorder(&context, true);
    const auto& device = devices.getOrImportDevice(declaration);

    CHECK_EQ(device.getDeclaration(), declaration);
    CHECK_EQ(device.getDeclaration(), device.getDefinition());
  }

  SUBCASE("import device") {
    const auto declaration = declarations.lookup<DeviceOp>("device");
    DiagRecorder recorder(&context, true);
    const auto& device = devices.getOrImportDevice(declaration);

    CHECK_EQ(device.getDeclaration(), declaration);
    CHECK_NE(device.getDeclaration(), device.getDefinition());
    CHECK_EQ(device.getDefinition()->getParentOp(), nullptr);
  }
}

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

TEST_CASE("mlir::ktdf_arch::DeviceManager persistence") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  // Construct and parse the test MLIR program.
  auto module = parse(&context, INPUTS_DIR "device-manager-test.mlir");
  SymbolTable declarations(module.get());

  PassManager pm(&context);
  pm.addPass(std::make_unique<UseDeviceManagerPass>());
  pm.addPass(std::make_unique<UseDeviceManagerPass>());
  pm.addNestedPass<func::FuncOp>(std::make_unique<UseDeviceManagerPass>());
  pm.addPass(std::make_unique<UseDeviceManagerPass>());
  pm.addNestedPass<func::FuncOp>(std::make_unique<UseDeviceManagerPass>());

  CHECK(succeeded(pm.run(module.get())));
}
