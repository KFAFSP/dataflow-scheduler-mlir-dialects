set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KTDFArch)
add_dataflow_scheduler_dialects_tablegen_target(DataflowSchedulerDialectsKTDFArchTransformsPassesIncGen)

add_dataflow_scheduler_dialects_doc(Passes KTDFArchPasses ./ -gen-pass-doc)
