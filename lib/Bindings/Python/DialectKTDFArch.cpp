//===-- DialectKTDFArch.cpp -------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/Attributes.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>

#include "Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
namespace nb = nanobind;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdf_arch {

DECLARE_SINGLETON_TYPE(ktdf_arch, MemoryType);
DECLARE_SINGLETON_TYPE(ktdf_arch, ExecutionUnitType);
DECLARE_SINGLETON_TYPE(ktdf_arch, PortType);

struct PyMapAttr : PyConcreteAttribute<PyMapAttr> {
  static auto isaFunction(MlirAttribute attr) -> bool {
    return isa<mlir::ktdf_arch::MapAttr>(unwrap(attr));
  }
  static auto getTypeIdFunction() -> MlirTypeID {
    return wrap(mlir::ktdf_arch::MapAttr::getTypeID());
  }
  static constexpr const char* pyClassName = "MapAttr";

  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](nb::typed<nb::dict, MlirAttribute, MlirAttribute> dict,
           DefaultingPyMlirContext context) {
          llvm::SmallVector<mlir::ktdf_arch::MapAttr::Entry> entries;
          for (auto [k, v] : dict) {
            entries.emplace_back(unwrap(nb::cast<MlirAttribute>(k)),
                                 unwrap(nb::cast<MlirAttribute>(v)));
          }
          return PyMapAttr(context->getRef(),
                           wrap(mlir::ktdf_arch::MapAttr::get(
                               unwrap(context->get()), entries)));
        },
        nb::arg("dict"), nb::arg("context").none() = nb::none());

    c.def(
        "getAttr",
        [](PyMapAttr self, MlirAttribute key) {
          return wrap(cast<mlir::ktdf_arch::MapAttr>(unwrap(self))
                          .getAttr(unwrap(key)));
        },
        nb::arg("key"));

    c.def("__contains__", [](PyMapAttr self, MlirAttribute key) {
      return cast<mlir::ktdf_arch::MapAttr>(unwrap(self))
                 .getAttr(unwrap(key)) != nullptr;
    });
    c.def("__contains__", [](PyMapAttr self, nb::handle) { return false; });

    c.def("__len__", [](PyMapAttr self) {
      return cast<mlir::ktdf_arch::MapAttr>(unwrap(self)).size();
    });

    c.def("__getitem__", [](PyMapAttr self, MlirAttribute key) {
      const auto result =
          cast<mlir::ktdf_arch::MapAttr>(unwrap(self)).getAttr(unwrap(key));
      if (!result) {
        throw nb::key_error();
      }
      return wrap(result);
    });

    c.def("__iter__", [](PyMapAttr self) {
      const auto attr = cast<mlir::ktdf_arch::MapAttr>(unwrap(self));
      const auto keys =
          llvm::map_range(attr.getEntries(),
                          [](std::pair<mlir::Attribute, mlir::Attribute> pair) {
                            return wrap(pair.first);
                          });
      return nb::make_iterator(nb::type<PyMapAttr>(), "KeyIterator",
                               keys.begin(), keys.end());
    });

    c.def("items",
          [](PyMapAttr self)
              -> nb::typed<nb::dict, MlirAttribute, MlirAttribute> {
            nb::dict result;
            for (auto [key, value] :
                 cast<mlir::ktdf_arch::MapAttr>(unwrap(self))) {
              result[nb::cast(wrap(key))] = nb::cast(wrap(value));
            }
            return result;
          });
  }
};

}  // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdf_arch

NB_MODULE(_dataflow_scheduler_dialects_ktdf_arch, m) {
  m.def("register_dialect", [](MlirDialectRegistry registry) {
    unwrap(registry)->insert<mlir::ktdf_arch::KTDFArchDialect>();
  });

  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdf_arch;

  PyMemoryType::bind(m);
  PyExecutionUnitType::bind(m);
  PyPortType::bind(m);

  PyMapAttr::bind(m);
}
