// RUN: dataflow-scheduler-dialects-opt --canonicalize %s -allow-unregistered-dialect | FileCheck %s

// Canonicalization of the offset maps and sizes of a `ktdf.data_transfer`.

// An applied source index folds into the source map.
// CHECK-LABEL: func.func @apply_source(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %{{.+}}: memref<64xf16>, %[[I:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[I]] * 2 + 1] size [64]
func.func @apply_source(%src: memref<128xf16>, %dst: memref<64xf16>, %i: index) {
  %c0 = arith.constant 0 : index
  %index = affine.apply affine_map<(d0) -> (d0 * 2 + 1)>(%i)
  ktdf.data_transfer from %src[%index] size [64] to %dst[%c0] size [64] : memref<128xf16>, memref<64xf16>
  return
}

// An applied destination index folds into the destination map.
// CHECK-LABEL: func.func @apply_destination(
// CHECK-SAME:      %{{.+}}: memref<64xf16>, %[[DST:.+]]: memref<128xf16>, %[[I:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from {{.+}} to %[[DST]][%[[I]] + 64] size [64]
func.func @apply_destination(%src: memref<64xf16>, %dst: memref<128xf16>, %i: index) {
  %c0 = arith.constant 0 : index
  %index = affine.apply affine_map<(d0) -> (d0 + 64)>(%i)
  ktdf.data_transfer from %src[%c0] size [64] to %dst[%index] size [64] : memref<64xf16>, memref<128xf16>
  return
}

// Applied indices on both sides fold into their maps.
// CHECK-LABEL: func.func @apply_both(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %[[DST:.+]]: memref<128xf16>, %[[I:.+]]: index, %[[J:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[I]] * 2] size [64] to %[[DST]][%[[J]] + 1] size [64]
func.func @apply_both(%src: memref<128xf16>, %dst: memref<128xf16>, %i: index, %j: index) {
  %a = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %b = affine.apply affine_map<(d0) -> (d0 + 1)>(%j)
  ktdf.data_transfer from %src[%a] size [64] to %dst[%b] size [64] : memref<128xf16>, memref<128xf16>
  return
}

// Nested applies fold completely.
// CHECK-LABEL: func.func @apply_nested(
// CHECK-SAME:      %[[SRC:.+]]: memref<?xf16>, %{{.+}}: memref<64xf16>, %[[I:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[I]] * 6] size [64]
func.func @apply_nested(%src: memref<?xf16>, %dst: memref<64xf16>, %i: index) {
  %c0 = arith.constant 0 : index
  %a = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %b = affine.apply affine_map<(d0) -> (d0 * 3)>(%a)
  ktdf.data_transfer from %src[%b] size [64] to %dst[%c0] size [64] : memref<?xf16>, memref<64xf16>
  return
}

// The symbols of an apply become dims of the map.
// CHECK-LABEL: func.func @apply_symbol(
// CHECK-SAME:      %[[SRC:.+]]: memref<?xf16>, %{{.+}}: memref<64xf16>, %[[I:.+]]: index, %[[N:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[N]] + %[[I]]] size [64]
func.func @apply_symbol(%src: memref<?xf16>, %dst: memref<64xf16>, %i: index, %n: index) {
  %c0 = arith.constant 0 : index
  %index = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%n]
  ktdf.data_transfer from %src[%index] size [64] to %dst[%c0] size [64] : memref<?xf16>, memref<64xf16>
  return
}

// An applied index with other users folds, and the apply is kept for them.
// CHECK-LABEL: func.func @apply_shared(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %{{.+}}: memref<64xf16>, %[[I:.+]]: index
// CHECK:         %[[INDEX:.+]] = affine.apply
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[I]] * 2] size [64]
// CHECK:         "test.use"(%[[INDEX]])
func.func @apply_shared(%src: memref<128xf16>, %dst: memref<64xf16>, %i: index) {
  %c0 = arith.constant 0 : index
  %index = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  ktdf.data_transfer from %src[%index] size [64] to %dst[%c0] size [64] : memref<128xf16>, memref<64xf16>
  "test.use"(%index) : (index) -> ()
  return
}

// Constant indices fold into the map.
// CHECK-LABEL: func.func @constant_index(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %[[DST:.+]]: memref<64xf16>
// CHECK:         ktdf.data_transfer from %[[SRC]][4] size [64] to %[[DST]][0] size [64]
func.func @constant_index(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  ktdf.data_transfer from %src[%c4] size [64] to %dst[%c0] size [64] : memref<128xf16>, memref<64xf16>
  return
}

// An applied index into a memory side of a FIFO transfer folds into its map.
// CHECK-LABEL: func.func @apply_from_fifo(
// CHECK-SAME:      %{{.+}}: !ktdf.fifo.slot<{{.+}}>, %[[DST:.+]]: memref<128xf16>, %[[I:.+]]: index
// CHECK-NOT:     affine.apply
// CHECK:         ktdf.data_transfer from %{{.+}} size [64] to %[[DST]][%[[I]] * 2] size [64]
func.func @apply_from_fifo(%slot: !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, %dst: memref<128xf16>, %i: index) {
  %index = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [64] to %dst[%index] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}

// A transfer with canonical maps is kept.
// CHECK-LABEL: func.func @canonical(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %[[DST:.+]]: memref<128xf16>, %[[I:.+]]: index, %[[J:.+]]: index
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[I]]] size [64] to %[[DST]][%[[J]]] size [64]
func.func @canonical(%src: memref<128xf16>, %dst: memref<128xf16>, %i: index, %j: index) {
  ktdf.data_transfer from %src[%i] size [64] to %dst[%j] size [64] : memref<128xf16>, memref<128xf16>
  return
}

// Constant sizes fold into the static sizes.
// CHECK-LABEL: func.func @constant_size(
// CHECK:         ktdf.data_transfer from {{.+}} size [64] to {{.+}} size [64]
func.func @constant_size(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  ktdf.data_transfer from %src[%c0] size [%c64] to %dst[%c0] size [%c64] : memref<128xf16>, memref<64xf16>
  return
}

// Only the constant sizes of mixed sizes fold.
// CHECK-LABEL: func.func @mixed_size(
// CHECK-SAME:      %{{.+}}: memref<?x128xf16>, %{{.+}}: memref<?x64xf16>, %[[N:.+]]: index
// CHECK:         ktdf.data_transfer from {{.+}} size [%[[N]], 64] to {{.+}} size [%[[N]], 64]
func.func @mixed_size(%src: memref<?x128xf16>, %dst: memref<?x64xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  ktdf.data_transfer from %src[%c0, %c0] size [%n, %c64] to %dst[%c0, %c0] size [%n, %c64] : memref<?x128xf16>, memref<?x64xf16>
  return
}

// A constant size of a FIFO side folds into the static sizes.
// CHECK-LABEL: func.func @constant_size_fifo(
// CHECK:         ktdf.data_transfer from %{{.+}} size [64] to {{.+}} size [64]
func.func @constant_size_fifo(%slot: !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [%c64] to %dst[%c0] size [%c64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// Dynamic sizes are kept.
// CHECK-LABEL: func.func @dynamic_size(
// CHECK-SAME:      %{{.+}}: memref<?xf16>, %{{.+}}: memref<?xf16>, %[[N:.+]]: index
// CHECK:         ktdf.data_transfer from {{.+}} size [%[[N]]] to {{.+}} size [%[[N]]]
func.func @dynamic_size(%src: memref<?xf16>, %dst: memref<?xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  ktdf.data_transfer from %src[%c0] size [%n] to %dst[%c0] size [%n] : memref<?xf16>, memref<?xf16>
  return
}

// Negative constant sizes are kept.
// CHECK-LABEL: func.func @negative_size(
// CHECK:         %[[SIZE:.+]] = arith.constant -1 : index
// CHECK:         ktdf.data_transfer from {{.+}} size [%[[SIZE]]] to {{.+}} size [%[[SIZE]]]
func.func @negative_size(%src: memref<128xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %cm1 = arith.constant -1 : index
  ktdf.data_transfer from %src[%c0] size [%cm1] to %dst[%c0] size [%cm1] : memref<128xf16>, memref<128xf16>
  return
}
