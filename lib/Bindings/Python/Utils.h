//===-- Utils.h -------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECTS_BINDINGS_PYTHON_UTILS_H_
#define DATAFLOW_SCHEDULER_DIALECTS_BINDINGS_PYTHON_UTILS_H_

#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/CAPI/IR.h>
#include <nanobind/nanobind.h>

#define IMPORT_INT_ENUM_TYPECASTER(namespace, name)                         \
  template <>                                                               \
  struct nb::detail::type_caster<mlir::namespace ::name> {                  \
    NB_TYPE_CASTER(mlir::namespace ::name,                                  \
                   const_name(MAKE_MLIR_PYTHON_QUALNAME(                    \
                       "dialects." #namespace "." #name)))                  \
                                                                            \
    bool from_python(handle src, uint8_t, cleanup_list*) noexcept {         \
      std::underlying_type_t<mlir::namespace ::name> as_int;                \
      if (!nb::try_cast(src, as_int)) return false;                         \
      value = static_cast<mlir::namespace ::name>(as_int);                  \
      return true;                                                          \
    }                                                                       \
                                                                            \
    static handle from_cpp(mlir::namespace ::name src, rv_policy,           \
                           cleanup_list*) noexcept {                        \
      const auto as_int =                                                   \
          static_cast<std::underlying_type_t<mlir::namespace ::name>>(src); \
                                                                            \
      return nb::module_::import_(                                          \
                 MAKE_MLIR_PYTHON_QUALNAME("dialects." #namespace))         \
          .attr(#name)(as_int)                                              \
          .release();                                                       \
    }                                                                       \
  }

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktir {

template <class Derived, class Type>
struct PyCppSingletonTypeBase : PyConcreteType<Derived> {
  using Base = PyCppSingletonTypeBase<Derived, Type>;

  static auto isaFunction(MlirType type) -> bool {
    return isa<Type>(unwrap(type));
  }
  static auto getTypeIdFunction() -> MlirTypeID {
    return wrap(Type::getTypeID());
  }

  static void bindDerived(typename PyConcreteType<Derived>::ClassTy& c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return Derived(context->getRef(),
                         wrap(Type::get(unwrap(context.get()->get()))));
        },
        nanobind::arg("context").none() = nanobind::none());
  }

  using PyConcreteType<Derived>::PyConcreteType;
};

}  // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktir

#define DECLARE_SINGLETON_TYPE(namespace, name)                        \
  struct Py##name                                                      \
      : mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktir::              \
            PyCppSingletonTypeBase<Py##name, mlir::namespace ::name> { \
    using Base::Base;                                                  \
    static constexpr const char* pyClassName = #name;                  \
  }

#endif  // DATAFLOW_SCHEDULER_DIALECTS_BINDINGS_PYTHON_UTILS_H_
