// RUN: dataflow-scheduler-dialects-opt --show-dialects | FileCheck %s

// CHECK: Available Dialects:

// CHECK-DAG: agen
// CHECK-DAG: dataflow
// CHECK-DAG: ktdf
// CHECK-DAG: ktdf_arch
// CHECK-DAG: ktdf_lowering
// CHECK-DAG: symbol
// CHECK-DAG: uniform
// CHECK-DAG: vectorchain
