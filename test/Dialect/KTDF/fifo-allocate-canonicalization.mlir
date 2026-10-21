// RUN: dataflow-scheduler-dialects-opt --canonicalize %s -allow-unregistered-dialect | FileCheck %s

// Dropping of unused slots from a `ktdf.fifo.allocate`.

// An allocation without used slots is erased.
// CHECK-LABEL: func.func @all_unused(
// CHECK-NOT:     ktdf.fifo.allocate
// CHECK:         return
func.func @all_unused(%size: index) {
  %slots:2 = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
  return
}

// An unused static slot is dropped.
// CHECK-LABEL: func.func @unused_static(
// CHECK-NEXT:    %[[SLOT:.+]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
// CHECK-NEXT:    "test.use"(%[[SLOT]])
func.func @unused_static() {
  %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
  "test.use"(%slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) -> ()
  return
}

// An unused dynamic slot is dropped with its size.
// CHECK-LABEL: func.func @unused_dynamic_first(
// CHECK-SAME:      %{{.+}}: index, %[[USED:.+]]: index
// CHECK-NEXT:    %[[SLOT:.+]] = ktdf.fifo.allocate(%[[USED]]) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
// CHECK-NEXT:    "test.use"(%[[SLOT]])
func.func @unused_dynamic_first(%unused: index, %used: index) {
  %slots:2 = ktdf.fifo.allocate(%unused, %used) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
  "test.use"(%slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) -> ()
  return
}

// An unused dynamic slot after a used one is dropped with its size.
// CHECK-LABEL: func.func @unused_dynamic_last(
// CHECK-SAME:      %[[USED:.+]]: index, %{{.+}}: index
// CHECK-NEXT:    %[[SLOT:.+]] = ktdf.fifo.allocate(%[[USED]]) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
// CHECK-NEXT:    "test.use"(%[[SLOT]])
func.func @unused_dynamic_last(%used: index, %unused: index) {
  %slots:2 = ktdf.fifo.allocate(%used, %unused) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
  "test.use"(%slots#0) : (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) -> ()
  return
}

// An allocation with only used slots is kept.
// CHECK-LABEL: func.func @all_used(
// CHECK-SAME:      %[[SIZE:.+]]: index
// CHECK-NEXT:    %[[SLOTS:.+]]:2 = ktdf.fifo.allocate(%[[SIZE]]) -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
// CHECK-NEXT:    "test.use"(%[[SLOTS]]#0, %[[SLOTS]]#1)
func.func @all_used(%size: index) {
  %slots:2 = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
  "test.use"(%slots#0, %slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) -> ()
  return
}

// An unused private slot is dropped from the allocation and the private op.
// CHECK-LABEL: func.func @unused_private_slot(
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) {
// CHECK-NEXT:        %[[ALLOC:.+]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
// CHECK-NEXT:        ktdf.private_yield %[[ALLOC]]
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage {{.+}} {
// CHECK-NEXT:        "test.use"(%[[SLOT]])
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @unused_private_slot() {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) {
      %alloc:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
      ktdf.private_yield %alloc#0, %alloc#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "test.use"(%slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) -> ()
    }
  }
  return
}

// An unused dynamic private slot is dropped with its size.
// CHECK-LABEL: func.func @unused_private_dynamic_first(
// CHECK-SAME:      %{{.+}}: index, %[[USED:.+]]: index
// CHECK:         ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) {
// CHECK-NEXT:      %[[ALLOC:.+]] = ktdf.fifo.allocate(%[[USED]]) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
// CHECK-NEXT:      ktdf.private_yield %[[ALLOC]]
func.func @unused_private_dynamic_first(%unused: index, %used: index) {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) {
      %alloc:2 = ktdf.fifo.allocate(%unused, %used) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
      ktdf.private_yield %alloc#0, %alloc#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "test.use"(%slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>) -> ()
    }
  }
  return
}

// An allocation without used private slots is erased from the private op.
// CHECK-LABEL: func.func @unused_private_allocation(
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) {
// CHECK-NEXT:        %[[ALLOC:.+]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
// CHECK-NEXT:        ktdf.private_yield %[[ALLOC]]
// CHECK-NEXT:      }
func.func @unused_private_allocation() {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) {
      %unused = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %used = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
      ktdf.private_yield %unused, %used : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      "test.use"(%slots#1) : (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>) -> ()
    }
  }
  return
}
