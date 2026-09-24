add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFAnalysis
    StageDependency.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF/Analysis
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDF/Analysis

  GROUP DIALECT
  EXPORT_NAME KTDFAnalysis

  LINK_LIBS
    DataflowSchedulerDialectsKTDFDialect
)
