add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsSymbolDialect
    Symbol.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowSchedulerDialects_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Symbol
    ${DataflowSchedulerDialects_BINARY_DIR}/include/dataflow-scheduler/Dialect/Symbol

  GROUP DIALECT
  EXPORT_NAME SymbolDialect

  DEPENDS
    DataflowSchedulerDialectsSymbolIncGen

  LINK_MLIR_LIBS
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRLoopLikeInterface
)
