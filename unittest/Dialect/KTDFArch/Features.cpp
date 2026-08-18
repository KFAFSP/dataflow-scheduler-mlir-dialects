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

#include <doctest/doctest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Parser/Parser.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

using namespace mlir;
using namespace mlir::ktdf_arch;

namespace {

template <class FeatureAttr>
auto parseFeature(MLIRContext* context, StringRef value) {
  const auto source = Twine("\"parse.me\"() { value = ")
                          .concat(value)
                          .concat(" } : () -> ()")
                          .str();
  DiagRecorder recorder(context, true);
  ParserConfig config{context, true};
  auto parsed = parseSourceString<ModuleOp>(source, config);
  auto& op = parsed->getBody()->front();
  return cast<FeatureAttr>(op.getDiscardableAttr("value"));
}

template <class FeatureAttr>
auto testFeature(MLIRContext* context, StringRef provided, StringRef required)
    -> bool {
  const auto provide = parseFeature<FeatureAttr>(context, provided);
  const auto require = parseFeature<FeatureAttr>(context, required);
  return provide.test(require);
}

}  // namespace

TEST_CASE("mlir::ktdf_arch::feature::Load") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  SUBCASE("access_granularity") {
    CHECK(testFeature<feature::Load>(
        &context, "{ access_granularity = #ktdf_arch.map<\"A\" = []> }",
        "{ }"));
    CHECK(testFeature<feature::Load>(
        &context, "{ }",
        "{ access_granularity = #ktdf_arch.map<\"A\" = []> }"));
    CHECK(testFeature<feature::Load>(
        &context,
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 1}]> "
        "}",
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 1}]> "
        "}"));
    CHECK(testFeature<feature::Load>(
        &context, "{ }", "{ word_size = #ktdf_arch.map<\"A\" = 1> }"));
    CHECK(testFeature<feature::Load>(
        &context, "{ word_size = #ktdf_arch.map<> }",
        "{ word_size = #ktdf_arch.map<\"A\" = 1> }"));

    CHECK_FALSE(testFeature<feature::Load>(
        &context, "{ }",
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{}]> }"));
    CHECK_FALSE(testFeature<feature::Load>(
        &context,
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 8, "
        "align_in_words = 8}]> }",
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 8, "
        "align_in_words = 1}]> }"));
    CHECK_FALSE(testFeature<feature::Load>(
        &context,
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 1}]> "
        "}",
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 8}]> "
        "}"));

    CHECK(testFeature<feature::Load>(
        &context,
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 8, "
        "align_in_words = 4}]> }",
        "{ access_granularity = #ktdf_arch.map<\"A\" = [{size_in_words = 4, "
        "align_in_words = 8}]> }"));
  }
}

TEST_CASE("mlir::ktdf_arch::feature::SIMD") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  SUBCASE("splat") {
    CHECK(testFeature<feature::SIMD>(&context, "{ splat }", "{  }"));

    CHECK_FALSE(testFeature<feature::SIMD>(&context, "{ }", "{ splat }"));

    CHECK(testFeature<feature::SIMD>(&context, "{ splat }", "{ splat }"));
  }

  SUBCASE("zero_pad") {
    CHECK(testFeature<feature::SIMD>(&context, "{ zero_pad }", "{  }"));

    CHECK_FALSE(testFeature<feature::SIMD>(&context, "{ }", "{ zero_pad }"));

    CHECK(testFeature<feature::SIMD>(&context, "{ zero_pad }", "{ zero_pad }"));
  }

  SUBCASE("lanes") {
    CHECK(testFeature<feature::SIMD>(&context, "{ lanes = #ktdf_arch.map<> }",
                                     "{  }"));
    CHECK(testFeature<feature::SIMD>(&context, "{ }",
                                     "{ lanes = #ktdf_arch.map<> }"));
    CHECK(testFeature<feature::SIMD>(&context, "{ lanes = #ktdf_arch.map<> }",
                                     "{ lanes = #ktdf_arch.map<> }"));

    CHECK_FALSE(testFeature<feature::SIMD>(
        &context, "{ }", "{ lanes = #ktdf_arch.map<f16 = 2> }"));
    CHECK_FALSE(testFeature<feature::SIMD>(
        &context, "{ lanes = #ktdf_arch.map<f16 = 1> }",
        "{ lanes = #ktdf_arch.map<f16 = 2> }"));
    CHECK_FALSE(testFeature<feature::SIMD>(
        &context, "{ lanes = #ktdf_arch.map<f32 = 1> }",
        "{ lanes = #ktdf_arch.map<f16 = 2> }"));

    CHECK(testFeature<feature::SIMD>(&context,
                                     "{ lanes = #ktdf_arch.map<f16 = 4> }",
                                     "{ lanes = #ktdf_arch.map<f16 = 2> }"));
  }
}

TEST_CASE("mlir::ktdf_arch::feature::Queue") {
  // Setup an MLIR context.
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  SUBCASE("ordered") {
    CHECK(testFeature<feature::Queue>(&context, "{ ordered }", "{  }"));

    CHECK_FALSE(testFeature<feature::Queue>(&context, "{ }", "{ ordered }"));

    CHECK(testFeature<feature::Queue>(&context, "{ ordered }", "{ ordered }"));
  }

  SUBCASE("size") {
    CHECK(testFeature<feature::Queue>(&context, "{ size = 128 }", "{  }"));
    CHECK(testFeature<feature::Queue>(&context, "{ size = 128 }",
                                      "{ size = 128 }"));

    CHECK_FALSE(testFeature<feature::Queue>(&context, "{ }", "{ size = 128 }"));
    CHECK_FALSE(testFeature<feature::Queue>(&context, "{ size = 64 }",
                                            "{ size = 128 }"));

    CHECK(testFeature<feature::Queue>(&context, "{ size = 128 }",
                                      "{ size = 64 }"));
  }

  SUBCASE("depth") {
    CHECK(testFeature<feature::Queue>(&context, "{ depth = 128 }", "{  }"));
    CHECK(testFeature<feature::Queue>(&context, "{ depth = 128 }",
                                      "{ depth = 128 }"));

    CHECK_FALSE(
        testFeature<feature::Queue>(&context, "{ }", "{ depth = 128 }"));
    CHECK_FALSE(testFeature<feature::Queue>(&context, "{ depth = 64 }",
                                            "{ depth = 128 }"));

    CHECK(testFeature<feature::Queue>(&context, "{ depth = 128 }",
                                      "{ depth = 64 }"));
  }
}
