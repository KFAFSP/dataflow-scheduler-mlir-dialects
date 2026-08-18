//===-- KTDFArchAttributes.h ------------------------------------*- c++ -*-===//
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
// This file defines the attributes in the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>

#include <type_traits>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"  // IWYU pragma: keep

namespace mlir::ktdf_arch {

//===----------------------------------------------------------------------===//
// Attribute Constraints
//===----------------------------------------------------------------------===//

/// Named constraint for the I64Attr in TableGen.
struct I64Attr : IntegerAttr {
  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    const auto int_attr = dyn_cast<IntegerAttr>(attr);
    return int_attr && classof(int_attr);
  }
  [[nodiscard]] static auto classof(IntegerAttr attr) -> bool {
    return attr.getType().isSignlessInteger(64U);
  }

  using ValueType = int64_t;

  using IntegerAttr::IntegerAttr;

  static auto get(MLIRContext* context, int64_t value) -> I64Attr;

  [[nodiscard]] auto getValue() const -> int64_t {
    return IntegerAttr::getValue().getSExtValue();
  }
};

/// Named constraint for an attribute that stores a directed adjacency matrix.
struct AdjacencyMatrixAttr : ElementsAttr {
  /// Type that represents a directed adjacency pair.
  using Edge = std::pair<int64_t, int64_t>;

  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    const auto elements = dyn_cast<ElementsAttr>(attr);
    if (!elements || !elements.getElementType().isSignlessInteger(1)) {
      return false;
    }

    const auto shape = elements.getShapedType().getShape();
    return shape.size() == 2 && shape[0] == shape[1];
  }

  using ElementsAttr::ElementsAttr;

  /// Constructs an adjacency matrix that has the same value everywhere.
  static auto get(MLIRContext* context, int64_t dim, bool splat_value = true)
      -> AdjacencyMatrixAttr {
    return get(context, dim, ArrayRef<bool>{splat_value});
  }
  /// Constructs a dense adjacency matrix.
  static auto get(MLIRContext* context, int64_t dim, ArrayRef<bool> values)
      -> AdjacencyMatrixAttr;
  /// Constructs a sparse adjacency matrix.
  static auto get(MLIRContext* context, int64_t dim, ArrayRef<Edge> adjacencies)
      -> AdjacencyMatrixAttr;

  /// Tests whether @p edge is adjacent in the matrix.
  [[nodiscard]] auto contains(Edge edge) const -> bool {
    return contains(edge.first, edge.second);
  }
  /// Tests whether @p source is adjacent to @p target.
  [[nodiscard]] auto contains(int64_t source, int64_t target) const -> bool;

  /// Gets the number of nodes.
  [[nodiscard]] auto getDim() const -> int64_t {
    return getShapedType().getDimSize(0);
  }
};

}  // namespace mlir::ktdf_arch

/// Auto-generated includes.
#define GET_ATTRDEF_CLASSES
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h.inc"  // IWYU pragma: export

namespace mlir::ktdf_arch {

// Named constraint for an ArrayAttr with constrained value types.
template <class Value>
struct TypedArrayAttr : ArrayAttr {
  using ValueType = ArrayRef<Value>;

  static_assert(std::is_base_of_v<Attribute, Value>);

  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    const auto array_attr = dyn_cast<ArrayAttr>(attr);
    return array_attr && classof(array_attr);
  }
  [[nodiscard]] static auto classof(ArrayAttr attr) -> bool {
    return llvm::all_of(attr.getValue(), [](const auto& value) -> bool {
      return isa<Value>(value);
    });
  }

  using ArrayAttr::ArrayAttr;

  static auto get(MLIRContext* context, ArrayRef<Value> values) -> TypedAttr {
    return cast<TypedArrayAttr>(ArrayAttr::get(context, values));
  }

  [[nodiscard]] auto getValue() const -> ArrayRef<Value> {
    const auto base = ArrayAttr::getValue();
    return ArrayRef<Value>(reinterpret_cast<const Value*>(base.data()),
                           base.size());
  }
  template <class T = typename Value::ValueType>
  [[nodiscard]] auto getAsValueRange() const -> T {
    return ArrayAttr::getAsValueRange<Value>();
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = Value;
  using iterator = typename ArrayRef<Value>::iterator;

  [[nodiscard]] auto begin() const -> iterator { return getValue().begin(); }
  [[nodiscard]] auto end() const -> iterator { return getValue().end(); }
};

/// Named constraint for a MapAttr with constrained key and value types.
template <class Key, class Value>
struct TypedMapAttr : MapAttr {
  /// Type of the map entries.
  using Entry = std::pair<Key, Value>;

  static_assert(std::is_base_of_v<Attribute, Key>);
  static_assert(std::is_base_of_v<Attribute, Value>);

  [[nodiscard]] static auto classof(Attribute attr) -> bool {
    const auto map_attr = dyn_cast<MapAttr>(attr);
    return map_attr && classof(map_attr);
  }
  [[nodiscard]] static auto classof(MapAttr attr) -> bool {
    return llvm::all_of(attr.getEntries(), [](const auto& entry) -> bool {
      return isa<Key>(entry.first) && isa<Value>(entry.second);
    });
  }

  using MapAttr::MapAttr;

  static auto get(MLIRContext* context, ArrayRef<Entry> entries)
      -> TypedMapAttr {
    return cast<TypedMapAttr>(MapAttr::get(
        context, ArrayRef<MapAttr::Entry>(
                     reinterpret_cast<const MapAttr::Entry*>(entries.data()),
                     entries.size())));
  }

  /// Gets the entries in the map.
  [[nodiscard]] auto getEntries() const -> ArrayRef<Entry> {
    const auto base = MapAttr::getEntries();
    return ArrayRef<Entry>(reinterpret_cast<const Entry*>(base.data()),
                           base.size());
  }

  /// Gets the attribute associated with @p key , if any.
  [[nodiscard]] auto getAttr(Key key) const -> Value {
    for (auto& entry : getEntries()) {
      if (entry.first == key) {
        return cast<Value>(entry.second);
      }
    }

    return nullptr;
  }

  /// Gets the attribute value associated with @p key , if any.
  template <class ValueType = typename Value::ValueType>
  [[nodiscard]] auto getValue(Key key) const -> std::optional<ValueType> {
    const auto attr = getAttr(key);
    if (attr) {
      return attr.getValue();
    }

    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = Entry;
  using iterator = typename ArrayRef<Entry>::iterator;

  [[nodiscard]] auto begin() const -> iterator { return getEntries().begin(); }
  [[nodiscard]] auto end() const -> iterator { return getEntries().end(); }
};

}  // namespace mlir::ktdf_arch

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_H_
