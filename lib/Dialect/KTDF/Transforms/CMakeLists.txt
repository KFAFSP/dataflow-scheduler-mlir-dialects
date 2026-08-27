add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFTransforms
    CodeMotion.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF/Transforms
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDF/Transforms

  GROUP DIALECT
  EXPORT_NAME KTDFTransforms

  LINK_LIBS
    DataflowSchedulerDialectsKTDFDialect
)
