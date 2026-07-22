add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsVectorChainDialect
    VectorChain.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/VectorChain
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/VectorChain

  GROUP DIALECT
  EXPORT_NAME VectorChainDialect

  DEPENDS
    DataflowSchedulerDialectsVectorChainIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect
  
  LINK_MLIR_LIBS
    MLIRIR
    MLIRSideEffectInterfaces
)
