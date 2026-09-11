add_dataflow_scheduler_dialects_tool(ktdf-archgen
    ktdf-archgen.cpp

  EXPORT_NAME ktdf-archgen
)
target_link_libraries(ktdf-archgen PRIVATE
  DataflowSchedulerDialectsKTDFArchDialect
  DataflowSchedulerDialectsKTDFArchTransforms
)

mlir_target_link_libraries(ktdf-archgen PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRPDLDialect
  MLIRSupport
  MLIRTransforms
  MLIROptLib
)
llvm_update_compile_flags(ktdf-archgen)
