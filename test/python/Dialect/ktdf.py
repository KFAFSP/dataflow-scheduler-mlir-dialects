# RUN: python %s

from mlir_scheduler.ir import Context, DialectRegistry, StringAttr
import mlir_scheduler.dialects.ktdf as ktdf

registry = DialectRegistry()
ktdf.register_dialect(registry)
ctx = Context()

with ctx:
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()

    # TokenType
    token_type = ktdf.TokenType.get()
    assert token_type.__str__() == "!ktdf.token"

    # FifoSlotType
    src = StringAttr.get("DDR")
    dest = StringAttr.get("SFU")
    fifo_slot_type = ktdf.FifoSlotType.get(src, dest, 42, token_type)
    assert fifo_slot_type.__str__() == '!ktdf.fifo.slot<"DDR" -> "SFU", 42x!ktdf.token>'
    assert fifo_slot_type.src == src
    assert fifo_slot_type.dest == dest
    assert fifo_slot_type.numElements == 42
    assert fifo_slot_type.elementType == token_type

    # LoopTypeAttr
    loop_type = ktdf.LoopTypeAttr.get(ktdf.LoopType.ParallelLoop)
    assert loop_type.__str__() == "#ktdf.loop_type<parallel_loop>"
    assert loop_type.value == ktdf.LoopType.ParallelLoop
