// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Round-trip: memref-to-fifo with non-identity source subscript.
// CHECK-LABEL: func.func @transfer_with_affine_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}} + 3, %{{.*}} * 2]
// CHECK-SAME: size [1, 64]
// CHECK-SAME: to %{{.*}} size [64]
func.func @transfer_with_affine_subscript() {
  %A = memref.alloc() : memref<128x128xf16, "DDR">
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
  %i = arith.constant 1 : index
  %j = arith.constant 2 : index
  ktdf.data_transfer
    from %A[%i + 3, %j * 2] size [1, 64]
    to %fifo size [64]
    : memref<128x128xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
  return
}

// Round-trip: fifo-to-memref with constant subscript (broadcast pattern).
// CHECK-LABEL: func.func @transfer_with_constant_subscript
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}} size [64]
// CHECK-SAME: to %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, 0]
// CHECK-SAME: size [1, 1, 1, 64]
func.func @transfer_with_constant_subscript(%a: index, %b: index, %c: index) {
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
  %B = memref.alloc() : memref<12x1x64x64xf16, "DDR">
  ktdf.data_transfer
    from %fifo size [64]
    to %B[%a, %b, %c, 0] size [1, 1, 1, 64]
    : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<12x1x64x64xf16, "DDR">
  return
}

// Round-trip: identity-form subscripts (existing pattern, preserved).
// CHECK-LABEL: func.func @transfer_identity
// CHECK:   ktdf.data_transfer
// CHECK-SAME: from %{{.*}}[%{{.*}}, %{{.*}}] size [1, 64]
// CHECK-SAME: to %{{.*}}[%{{.*}}] size [64]
func.func @transfer_identity(%i: index, %j: index, %c0: index) {
  %A = memref.alloc() : memref<128x128xf16, "DDR">
  %l1 = memref.alloc() : memref<64xf16, "L1">
  ktdf.data_transfer
    from %A[%i, %j] size [1, 64]
    to %l1[%c0] size [64]
    : memref<128x128xf16, "DDR">, memref<64xf16, "L1">
  return
}
