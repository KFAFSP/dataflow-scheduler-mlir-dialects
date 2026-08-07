// RUN: dataflow-scheduler-dialects-opt -allow-unregistered-dialect %s | dataflow-scheduler-dialects-opt -allow-unregistered-dialect | FileCheck %s

// Round trip test for ktdf.read_from_fifo

// CHECK-LABEL:   func.func @read_from_fifo_static() {
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:     %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]] : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf32>
// CHECK-NEXT:     %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[FIFO_1]] : <"SFU" -> "L1SU", 8xf32> -> tensor<8xf32>
// CHECK-NEXT:     "test.op"(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]]) : (tensor<64xf16>, tensor<8xf32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  func.func @read_from_fifo_static() {
    // Allocate a FIFO slot with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    
    // Read from the FIFO slot into a tensor
    %data0 = ktdf.read_from_fifo %slot0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
    
    // Allocate another FIFO slot with 8 f32 elements
    %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf32>
    
    // Read from the second FIFO slot
    %data1 = ktdf.read_from_fifo %slot1 : !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf32> -> tensor<8xf32>
    
    // Use the tensors
    "test.op"(%data0, %data1) : (tensor<64xf16>, tensor<8xf32>) -> ()
    
    return
  }
}

// CHECK-LABEL:   func.func @read_from_fifo_memref() {
// CHECK-NEXT:     %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:     %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]] : <"L1LU" -> "SFU", 64xf16> -> memref<64xf16>
// CHECK-NEXT:     "test.op"(%[[READ_FROM_FIFO_0]]) : (memref<64xf16>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @read_from_fifo_memref() {
    // Allocate a FIFO slot with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    // Read from the FIFO slot into an unrealized memref buffer
    %buf0 = ktdf.read_from_fifo %slot0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<64xf16>

    // Use the buffer
    "test.op"(%buf0) : (memref<64xf16>) -> ()

    return
  }
}