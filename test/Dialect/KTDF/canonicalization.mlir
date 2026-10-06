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
    %p:5 = ktdf.private -> (index, memref<f32>, memref<f64>, memref<f32>, !ktdf.token) {
      %mem = memref.alloc() : memref<f32>
      %unused = memref.alloc() : memref<f64>
      %tok = ktdf.create_token : !ktdf.token
      ktdf.private_yield %c0, %mem, %unused, %mem, %tok : index, memref<f32>, memref<f64>, memref<f32>, !ktdf.token
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

// CHECK-LABEL: func.func @combine_vias(
// CHECK-SAME:    %[[IN:.+]]: tensor<4xf16>
func.func @combine_vias(%in: tensor<4xf16>) -> tensor<4xf16> {
  // CHECK-NEXT: %[[OUT:.+]] = ktdf.via["A", "B", "C", "A"] %[[IN]]
  %hop1 = ktdf.via["A", "A", "B"] %in : tensor<4xf16>
  %hop2 = ktdf.via["B", "C"] %hop1 : tensor<4xf16>
  %hop3 = ktdf.via["A"] %hop2 : tensor<4xf16>
  // CHECK-NEXT: return %[[OUT]]
  return %hop3 : tensor<4xf16>
}
