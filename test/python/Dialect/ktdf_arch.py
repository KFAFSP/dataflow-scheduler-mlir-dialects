# RUN: python %s %test_source_root

import sys
from pathlib import Path

test_source_root = Path(sys.argv[1])

from mlir_scheduler.ir import (
    Context,
    DialectRegistry,
    IntegerAttr,
    IntegerType,
    Module,
    Operation,
    StringAttr,
    WalkOrder,
    WalkResult,
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

    def load_module() -> (
        tuple[
            Module, ktdf_arch.MemoryOp, ktdf_arch.ExecutionUnitOp, ktdf_arch.DatapathOp
        ]
    ):
        input_path = test_source_root / "Inputs" / "resource.mlir"
        input = Module.parseFile(str(input_path))
        memory: ktdf_arch.MemoryOp | None = None
        exec_unit: ktdf_arch.ExecutionUnitOp | None = None
        datapath: ktdf_arch.DatapathOp | None = None

        def visit(op: Operation) -> WalkResult:
            view = op.opview
            nonlocal memory, exec_unit, datapath
            if isinstance(view, ktdf_arch.MemoryOp):
                memory = view
            elif isinstance(view, ktdf_arch.ExecutionUnitOp):
                exec_unit = view
            elif isinstance(view, ktdf_arch.DatapathOp):
                datapath = view
            return WalkResult.ADVANCE

        input.operation.walk(visit, WalkOrder.POST_ORDER)

        assert memory is not None
        assert exec_unit is not None
        assert datapath is not None
        return (input, memory, exec_unit, datapath)

    module, memory, exec_unit, datapath = load_module()

    # Resource
    memory_res = ktdf_arch.Resource(memory)
    assert memory_res
    assert memory_res.kind == memory.kind
    assert memory_res.id == memory.id
    memory_res.id = None
    assert memory_res.id is None
    new_memory_id = StringAttr.get("new_memory_id")
    memory_res.id = new_memory_id
    assert memory.id == new_memory_id

    # Link
    assert exec_unit is not None
    assert datapath is not None
    datapath_link = ktdf_arch.Link(datapath)
    assert datapath_link
    assert datapath_link.sources == [datapath.source]
    assert datapath_link.targets == [datapath.target]
