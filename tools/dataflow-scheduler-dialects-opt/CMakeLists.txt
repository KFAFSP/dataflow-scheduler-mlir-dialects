# Get the names of all relevant exported DataflowSchedulerDialects targets.
get_property(_dialect_libs GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(_extension_libs GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)
get_property(_conversion_libs GLOBAL PROPERTY ${PROJECT_NAME}_CONVERSION_LIBS)

add_dataflow_scheduler_dialects_tool(dataflow-scheduler-dialects-opt
    dataflow-scheduler-dialects-opt.cpp

  EXPORT_NAME opt

  SUPPORT_PLUGINS
)
target_link_libraries(dataflow-scheduler-dialects-opt PRIVATE
  ${_dialect_libs}
  ${_extension_libs}
  ${_conversion_libs}
  DataflowSchedulerDialectsTestKTDF
)

mlir_target_link_libraries(dataflow-scheduler-dialects-opt PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(dataflow-scheduler-dialects-opt)
export_executable_symbols_for_plugins(dataflow-scheduler-dialects-opt)
