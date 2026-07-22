add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsAgenDialect
    Agen.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowSchedulerDialects_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Agen
    ${DataflowSchedulerDialects_BINARY_DIR}/include/dataflow-scheduler/Dialect/Agen

  GROUP DIALECT
  EXPORT_NAME AgenDialect

  DEPENDS
    DataflowSchedulerDialectsAgenIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::VectorChainDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRArithDialect
    # These are used by Utils.cpp, but do not require dialect loading.
    MLIRAffineDialect
    MLIRAffineAnalysis
)

