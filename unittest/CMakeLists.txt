enable_testing()
include(CTest)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY "https://github.com/doctest/doctest"
  GIT_TAG "6804767ee637789db8a5cb281381cae98dc36906" # v2.5.2
)
FetchContent_MakeAvailable(doctest)
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(dataflow-scheduler-dialects-tests
  Dialect/KTDFArch/Analysis/DeviceManager.cpp
  Dialect/KTDFArch/Analysis/NodeLinks.cpp
  Dialect/KTDFArch/Analysis/ResourceIds.cpp
  Dialect/KTDFArch/AdjacencyMatrixAttr.cpp
  Dialect/KTDFArch/Features.cpp
  Dialect/KTDFArch/Resource.cpp
  main.cpp
)

target_link_libraries(dataflow-scheduler-dialects-tests
  doctest::doctest
  DataflowSchedulerDialects::KTDFArchAnalysis
)
target_include_directories(dataflow-scheduler-dialects-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(dataflow-scheduler-dialects-tests PRIVATE
  "INPUTS_DIR=\"${PROJECT_SOURCE_DIR}/test/Inputs/\""
)
llvm_update_compile_flags(dataflow-scheduler-dialects-tests)
if(MLIR_LINK_MLIR_DYLIB)
  target_link_libraries(dataflow-scheduler-dialects-tests
    MLIR
  )
else()
  target_link_libraries(dataflow-scheduler-dialects-tests
    MLIRPass
    MLIRFuncDialect
  )
endif()

doctest_discover_tests(dataflow-scheduler-dialects-tests)

add_custom_target(dataflow-scheduler-dialects-ctest
  COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS dataflow-scheduler-dialects-tests
)
add_dependencies(check-dataflow-scheduler-dialects 
  dataflow-scheduler-dialects-ctest
)
