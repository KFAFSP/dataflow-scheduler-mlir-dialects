add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsTestKTDF
    TestDataMovementCoalescing.cpp

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFTransforms

  LINK_MLIR_LIBS
    MLIRIR
    MLIRPass
    MLIRTensorDialect
    MLIRTransformUtils
)
