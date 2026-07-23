add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFArchAnalysis
    DeviceManager.cpp
    NodeEndpoints.cpp
    NodeLinks.cpp
    ResourceIds.cpp
    ResourceKinds.cpp

  GROUP DIALECT
  EXPORT_NAME KTDFArchAnalysis
  
  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDFArch/Analysis
    ${PROJECT_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDFArch/Analysis

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFArchDialect
)

mlir_target_link_libraries(DataflowSchedulerDialectsKTDFArchAnalysis PUBLIC
  MLIRIR
  MLIRParser
  MLIRAnalysis
  MLIRPass
)
