add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsUniformDialect
    Uniform.cpp
  
  ADDITIONAL_HEADER_DIRS
    ${DataflowSchedulerDialects_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Uniform
    ${DataflowSchedulerDialects_BINARY_DIR}/include/dataflow-scheduler/Dialect/Uniform

  GROUP DIALECT
  EXPORT_NAME UniformDialect

  DEPENDS
    DataflowSchedulerDialectsUniformIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect

  LINK_MLIR_LIBS
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRSideEffectInterfaces
)
