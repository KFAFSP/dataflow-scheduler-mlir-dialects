add_library(dataflow-scheduler-dialects-tests-exemplar
  UseDeviceManagerPass.cpp
)
target_link_libraries(dataflow-scheduler-dialects-tests-exemplar PUBLIC
  DataflowSchedulerDialects::KTDFArchAnalysis
)
target_include_directories(dataflow-scheduler-dialects-tests-exemplar PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
llvm_update_compile_flags(dataflow-scheduler-dialects-tests-exemplar)
if(MLIR_LINK_MLIR_DYLIB)
  target_link_libraries(dataflow-scheduler-dialects-tests-exemplar PUBLIC
    MLIR
  )
else()
  target_link_libraries(dataflow-scheduler-dialects-tests-exemplar PUBLIC
    MLIRPass
    MLIRFuncDialect
  )
endif()
