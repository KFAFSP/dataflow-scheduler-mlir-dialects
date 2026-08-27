// RUN: dataflow-scheduler-dialects-opt --canonicalize %s -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @erase_empty_private(
func.func @erase_empty_private() {
  // CHECK: ktdf.pipeline
  ktdf.pipeline {
    // CHECK-NOT: ktdf.private
    ktdf.private {
      ktdf.private_yield
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "unregistered.op"() : () -> ()
    }
  }
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @canonicalize_private_results(
func.func @canonicalize_private_results() {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: ktdf.pipeline
  ktdf.pipeline {
    // CHECK: %[[P:.+]] = ktdf.private -> (memref<f32>) {
    %p:4 = ktdf.private -> (index, memref<f32>, memref<f64>, memref<f32>) {
      %mem = memref.alloc() : memref<f32>
      %unused = memref.alloc() : memref<f64>
      ktdf.private_yield %c0, %mem, %unused, %mem : index, memref<f32>, memref<f64>, memref<f32>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: "unregistered.op"(%[[C0]], %[[P]], %[[P]])
      "unregistered.op"(%p#0, %p#1, %p#3) : (index, memref<f32>, memref<f32>) -> ()
    }
  }
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @erase_empty_stage(
func.func @erase_empty_stage() {
  // CHECK: ktdf.pipeline
  ktdf.pipeline {
    // CHECK: ktdf.private
    ktdf.private {
      "unregistered.op"() : () -> ()
      ktdf.private_yield
    }
    // CHECK-NOT: ktdf.stage
    ktdf.stage depends_in(none) depends_out(none) {

    }
  }
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @erase_empty_pipeline(
func.func @erase_empty_pipeline() {
  // CHECK-NOT: ktdf.pipeline
  ktdf.pipeline {
    ktdf.private {
      ktdf.private_yield
    }
    ktdf.stage depends_in(none) depends_out(none) {

    }
  }
  // CHECK: return
  return
}
