# RUN: python %s

from mlir_scheduler.ir import (
    Context,
    DialectRegistry,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
import mlir_scheduler.dialects.ktdf_arch as ktdf_arch

registry = DialectRegistry()
ktdf_arch.register_dialect(registry)
ctx = Context()


def make_int(x: int) -> IntegerAttr:
    return IntegerAttr.get(IntegerType.get(64), x)


with ctx:
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()

    # MemoryType
    memory_type = ktdf_arch.MemoryType.get()
    assert memory_type.__str__() == "!ktdf_arch.memory"

    # ExecutionUnitType
    memory_type = ktdf_arch.ExecutionUnitType.get()
    assert memory_type.__str__() == "!ktdf_arch.exec_unit"

    # PortType
    memory_type = ktdf_arch.PortType.get()
    assert memory_type.__str__() == "!ktdf_arch.port"

    # MapAttr
    entries = {StringAttr.get("a"): make_int(1), make_int(2): StringAttr.get("b")}
    map_attr = ktdf_arch.MapAttr.get(entries)
    assert len(map_attr) == 2
    assert StringAttr.get("a") in map_attr
    assert map_attr.getAttr(StringAttr.get("a")) == make_int(1)
    assert make_int(2) in map_attr
    assert map_attr.getAttr(make_int(2)) == StringAttr.get("b")
    assert make_int(1) not in map_attr
    assert list(map_attr) == list(entries.keys())
    assert map_attr.items() == entries
