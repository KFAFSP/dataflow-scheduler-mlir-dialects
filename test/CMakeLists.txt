llvm_canonicalize_cmake_booleans(
  DataflowSchedulerDialects_ENABLE_PYTHON_BINDINGS
  DataflowSchedulerDialects_BUILD_TOOLS
  BUILD_SHARED_LIBS
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DataflowSchedulerDialects_TEST_DEPENDS
  dataflow-scheduler-dialects-opt
  ktdf-archgen
  FileCheck 
  not
)

if(DataflowSchedulerDialects_ENABLE_PYTHON_BINDINGS)
  list(APPEND DataflowSchedulerDialects_TEST_DEPENDS DataflowSchedulerDialectsPythonModules)
endif()

add_lit_testsuite(check-dataflow-scheduler-dialects
    "Running the scheduler dialect regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DataflowSchedulerDialects_TEST_DEPENDS}
)
set_target_properties(check-dataflow-scheduler-dialects PROPERTIES FOLDER "Tests")

add_lit_testsuites(dataflow-scheduler-dialects
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${DataflowSchedulerDialects_TEST_DEPENDS}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)
