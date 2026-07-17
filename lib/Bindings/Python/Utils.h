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
#include <mlir-c/Interfaces.h>
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

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktir {

// Backport of https://github.com/llvm/llvm-project/pull/181522
template <class Interface>
class PyConcreteOpInterface {
 protected:
  using ClassTy = nanobind::class_<Interface>;
  using GetTypeIDFunctionTy = MlirTypeID (*)();

 public:
  PyConcreteOpInterface(nanobind::object object,
                        DefaultingPyMlirContext context)
      : obj(std::move(object)) {
    if (!nanobind::try_cast<PyOperation*>(obj, operation)) {
      PyOpView* opview;
      if (nanobind::try_cast<PyOpView*>(obj, opview)) {
        operation = &opview->getOperation();
      };
    }

    if (operation != nullptr) {
      if (!mlirOperationImplementsInterface(*operation,
                                            Interface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nanobind::value_error((msg + Interface::pyClassName).c_str());
      }

      MlirIdentifier identifier = mlirOperationGetName(*operation);
      MlirStringRef stringRef = mlirIdentifierStr(identifier);
      opName = std::string(stringRef.data, stringRef.length);
    } else {
      if (!nanobind::try_cast<std::string>(obj.attr("OPERATION_NAME"), opName))
        throw nanobind::type_error(
            "Op interface does not refer to an operation or OpView class");

      if (!mlirOperationImplementsInterfaceStatic(
              mlirStringRefCreate(opName.data(), opName.length()),
              context.resolve().get(), Interface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nanobind::value_error((msg + Interface::pyClassName).c_str());
      }
    }
  }

  static void bind(nanobind::module_& m) {
    nanobind::class_<Interface> cls(m, Interface::pyClassName);
    cls.def(nanobind::init<nanobind::object, DefaultingPyMlirContext>(),
            nanobind::arg("object"),
            nanobind::arg("context") = nanobind::none(),
            "Creates an interface from a given operation/opview object or from "
            "a subclass of OpView. Raises ValueError if the operation does not "
            "implement the interface.")
        .def_prop_ro(
            "operation", &PyConcreteOpInterface::getOperationObject,
            "Returns an Operation for which the interface was constructed.")
        .def_prop_ro("opview", &PyConcreteOpInterface::getOpView,
                     "Returns an OpView subclass _instance_ for which the "
                     "interface was constructed");
    Interface::bindDerived(cls);
  }

  static void bindDerived(ClassTy& cls) {}

  bool isStatic() { return operation == nullptr; }

  nanobind::typed<nanobind::object, PyOperation> getOperationObject() {
    if (operation == nullptr)
      throw nanobind::type_error(
          "Cannot get an operation from a static interface");
    return operation->getRef().releaseObject();
  }

  nanobind::typed<nanobind::object, PyOpView> getOpView() {
    if (operation == nullptr)
      throw nanobind::type_error(
          "Cannot get an opview from a static interface");
    return operation->createOpView();
  }

  const std::string& getOpName() { return opName; }

 private:
  PyOperation* operation = nullptr;
  std::string opName;
  nanobind::object obj;
};

}  // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktir

#endif  // DATAFLOW_SCHEDULER_DIALECTS_BINDINGS_PYTHON_UTILS_H_
