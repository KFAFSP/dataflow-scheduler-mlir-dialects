//===-- DialectKTDF.cpp -----------------------------------------*- c++ -*-===//
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

#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/CAPI/IR.h>
#include <nanobind/nanobind.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"

using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
namespace nb = nanobind;

template <>
struct nb::detail::type_caster<mlir::ktdf::LoopType> {
  NB_TYPE_CASTER(mlir::ktdf::LoopType, const_name(MAKE_MLIR_PYTHON_QUALNAME(
                                           "dialects.ktdf.LoopType")))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
    std::underlying_type_t<mlir::ktdf::LoopType> as_int;
    if (!nb::try_cast(src, as_int)) return false;
    value = static_cast<mlir::ktdf::LoopType>(as_int);
    return true;
  }

  static handle from_cpp(mlir::ktdf::LoopType src, rv_policy,
                         cleanup_list*) noexcept {
    const auto as_int =
        static_cast<std::underlying_type_t<mlir::ktdf::LoopType>>(src);

    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("dialects.ktdf"))
        .attr("LoopType")(as_int)
        .release();
  }
};

struct PyTokenType : PyConcreteType<PyTokenType> {
  static auto isaFunction(MlirType type) -> bool {
    return isa<ktdf::TokenType>(unwrap(type));
  }
  static auto getTypeIdFunction() -> MlirTypeID {
    return wrap(ktdf::TokenType::getTypeID());
  }
  static constexpr const char* pyClassName = "TokenType";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyTokenType(
              context->getRef(),
              wrap(ktdf::TokenType::get(unwrap(context.get()->get()))));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyFifoSlotType : PyConcreteType<PyFifoSlotType> {
  static auto isaFunction(MlirType type) -> bool {
    return isa<ktdf::FifoSlotType>(unwrap(type));
  }
  static auto getTypeIdFunction() -> MlirTypeID {
    return wrap(ktdf::FifoSlotType::getTypeID());
  }
  static constexpr const char* pyClassName = "FifoSlotType";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](MlirAttribute src, MlirAttribute dest, int64_t numElements,
           MlirType elementType) {
          return PyFifoSlotType(
              PyMlirContext::forContext(mlirTypeGetContext(elementType)),
              wrap(ktdf::FifoSlotType::get(
                  unwrap(mlirTypeGetContext(elementType)), unwrap(src),
                  unwrap(dest), numElements, unwrap(elementType))));
        },
        nb::arg("src"), nb::arg("dest"), nb::arg("numElements"),
        nb::arg("elementType"));

    c.def_prop_ro("src", [](PyFifoSlotType self) {
      return wrap(cast<ktdf::FifoSlotType>(unwrap(self)).getSrc());
    });

    c.def_prop_ro("dest", [](PyFifoSlotType self) {
      return wrap(cast<ktdf::FifoSlotType>(unwrap(self)).getDest());
    });

    c.def_prop_ro("numElements", [](PyFifoSlotType self) {
      return cast<ktdf::FifoSlotType>(unwrap(self)).getNumElements();
    });

    c.def_prop_ro("elementType", [](PyFifoSlotType self) {
      return wrap(cast<ktdf::FifoSlotType>(unwrap(self)).getElementType());
    });
  }
};

struct PyLoopTypeAttr : PyConcreteAttribute<PyLoopTypeAttr> {
  static auto isaFunction(MlirAttribute type) -> bool {
    return isa<ktdf::LoopTypeAttr>(unwrap(type));
  }
  static auto getTypeIdFunction() -> MlirTypeID {
    return wrap(ktdf::LoopTypeAttr::getTypeID());
  }
  static constexpr const char* pyClassName = "LoopTypeAttr";
  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](ktdf::LoopType value, DefaultingPyMlirContext context) {
          return PyLoopTypeAttr(
              context->getRef(),
              wrap(ktdf::LoopTypeAttr::get(unwrap(context->get()), value)));
        },
        nb::arg("value"), nb::arg("context").none() = nb::none());

    c.def_prop_ro("value", [](PyLoopTypeAttr self) {
      return cast<ktdf::LoopTypeAttr>(unwrap(self)).getValue();
    });
  }
};

NB_MODULE(_dataflow_scheduler_dialects_ktdf, m) {
  m.def("register_dialect", [](MlirDialectRegistry registry) {
    unwrap(registry)->insert<ktdf::KTDFDialect>();
  });

  PyTokenType::bind(m);
  PyFifoSlotType::bind(m);

  PyLoopTypeAttr::bind(m);
}
