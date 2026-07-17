//===-- KTDFArchDialect.cpp -------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// KTDFArchOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {

struct KTDFArchOpAsmDialectInterface : OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  auto getAlias(Attribute attr, raw_ostream& os) const -> AliasResult override {
    if (const auto dict = dyn_cast<DictionaryAttr>(attr); dict) {
      const auto kind = dict.getAs<StringAttr>("kind");
      if (kind) {
        os << kind.strref();
        return AliasResult::FinalAlias;
      }
    }

    return AliasResult::NoAlias;
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// KTDFArchFeatureDialectInterface
//===----------------------------------------------------------------------===//

namespace {

struct KTDFArchFeatureDialectInterface : FeatureDialectInterface {
  KTDFArchFeatureDialectInterface(Dialect* dialect,
                                  const AttrVerifyMap& feature_verifiers,
                                  const FeatureTestMap& feature_tests)
      : FeatureDialectInterface(dialect),
        feature_verifiers_(feature_verifiers),
        feature_tests_(feature_tests) {}

  auto verify(Operation* op, const Feature& feature) const
      -> LogicalResult override {
    const auto impl = feature_verifiers_.lookup(feature.getName());
    if (!impl) {
      auto diag = op->emitError("unrecognized '")
                  << mlir::ktdf_arch::KTDFArchDialect::getDialectNamespace()
                  << "' feature: " << feature.getName();
      auto& note = diag.attachNote() << "expected one of: ";
      llvm::interleaveComma(feature_verifiers_, note,
                            [&](auto& entry) { note << entry.first.strref(); });
      return diag;
    }

    return impl(op, feature);
  }

  [[nodiscard]] auto test(Attribute provided, const Feature& required) const
      -> bool override {
    const auto impl = feature_tests_.lookup(required.getName());
    assert(impl);
    return impl(provided, required);
  }

 private:
  const AttrVerifyMap& feature_verifiers_;
  const FeatureTestMap& feature_tests_;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Feature
//===----------------------------------------------------------------------===//

auto Feature::verify(Operation* op) const -> LogicalResult {
  if (auto const* iface =
          dyn_cast_if_present<FeatureDialectInterface>(getNameDialect());
      iface) {
    return iface->verify(op, *this);
  }

  return success();
}

auto Feature::matchedBy(Attribute provided) const -> bool {
  if (provided == getValue() || (provided && isa<UnitAttr>(getValue()))) {
    return true;
  }

  if (auto const* iface =
          dyn_cast_if_present<FeatureDialectInterface>(getNameDialect());
      iface) {
    return iface->test(provided, *this);
  }

  return false;
}

auto Feature::matchedBy(Operation* op) const -> bool {
  return getFeature(op, *this).has_value();
}

//===----------------------------------------------------------------------===//
// KTDFArchDialect
//===----------------------------------------------------------------------===//

auto KTDFArchDialect::verifyOperationAttribute(Operation* op,
                                               NamedAttribute attr)
    -> LogicalResult {
  const auto verifier = property_verifiers_.lookup(attr.getName());
  if (!verifier) {
    auto diag = op->emitError("unrecognized '")
                << getDialectNamespace() << "' intrisic: " << attr.getName();
    auto& note = diag.attachNote() << "expected one of: ";
    llvm::interleaveComma(property_verifiers_, note,
                          [&](auto& entry) { note << entry.first.strref(); });
    return diag;
  }

  return verifier(op, attr);
}

void KTDFArchDialect::initialize() {
  registerAttributes();
  registerOps();
  registerTypes();
  registerIntrinsics();

  addInterfaces<KTDFArchOpAsmDialectInterface>();
  addInterface<KTDFArchFeatureDialectInterface>(feature_verifiers_,
                                                feature_tests_);
}
