# Test-only libraries, linked into dataflow-scheduler-dialects-opt only.

if(NOT DataflowSchedulerDialects_BUILD_TOOLS)
  # Only build these when dataflow-scheduler-dialects-opt is built.
  set(EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(Dialect)
