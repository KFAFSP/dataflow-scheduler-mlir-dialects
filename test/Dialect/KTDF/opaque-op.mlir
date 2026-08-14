// RUN: dataflow-scheduler-dialects-opt -allow-unregistered-dialect %s | dataflow-scheduler-dialects-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @pure(
// CHECK-SAME: %[[ARG0:.+]]:
func.func @pure(%arg0: tensor<64xf16>) -> tensor<64xf16> {
  // CHECK: %[[C1:.+]] = arith.constant {{{.+}}} dense<1.000000e+00>
  %c1 = arith.constant {ktdf_arch.maps_to = "SFU_REG"} dense<1.0> : tensor<64xf16>
  // CHECK: %[[EPS:.+]] = arith.constant 7.629390e-06
  %eps = arith.constant 0x36fffff6 : f32

  // CHECK: %[[OUT:.+]] = tensor.empty
  %out = tensor.empty() : tensor<64xf16>

  // CHECK: %[[PURE:.+]] = ktdf.opaque "PURE"
  // CHECK-NEXT: ins(%[[C1]], %[[ARG0]], %[[EPS]]:
  // CHECK-NEXT: outs(%[[OUT]]:
  %0 = ktdf.opaque "PURE" -> tensor<64xf16>
    ins(%c1, %arg0, %eps: tensor<64xf16>, tensor<64xf16>, f32)
    outs(%out: tensor<64xf16>)

  // CHECK: return %[[PURE]]
  return %0 : tensor<64xf16>
}

memref.global @c1 : memref<64xf16> = dense<1.0>

// CHECK-LABEL: func.func @side_effecting(
// CHECK-SAME: %[[ARG0:.+]]: memref<64xf16, "L1">,
// CHECK-SAME: %[[ARG1:.+]]: memref<64xf16, "L1">
func.func @side_effecting(%arg0: memref<64xf16, "L1">, %arg1: memref<64xf16, "L1">) {
  %c1 = memref.alloc() : memref<64xf16, "SFU_REG">
  %c1_init = memref.get_global @c1 : memref<64xf16>
  // CHECK: memref.copy %{{.+}}, %[[C1:.+]] :
  memref.copy %c1_init, %c1 : memref<64xf16> to memref<64xf16, "SFU_REG">

  // CHECK: %[[EPS:.+]] = arith.constant 7.629390e-06
  %eps = arith.constant 0x36fffff6 : f32

  // CHECK: ktdf.opaque "SIDE_EFFECTING"
  // CHECK-NEXT: ins(%[[C1]], %[[ARG0]], %[[EPS]]:
  // CHECK-NEXT: outs(%[[ARG1]]:
  ktdf.opaque "SIDE_EFFECTING"
    ins(%c1, %arg0, %eps: memref<64xf16, "SFU_REG">, memref<64xf16, "L1">, f32)
    outs(%arg1: memref<64xf16, "L1">)
  return
}
