// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @select_dynamic(
// CHECK-SAME:      %arg0: vector<1024xi8>, %arg1: vector<256xi2>) -> vector<256xi8> {
// CHECK-NEXT:      %0 = vectorchain.select_dynamic %arg0, %arg1 : vector<1024xi8>, vector<256xi2>, vector<256xi8>
// CHECK-NEXT:      return %0 : vector<256xi8>
// CHECK-NEXT:    }

module {
  func.func @select_dynamic(%activation: vector<1024xi8>, %ker_idx: vector<256xi2>) -> vector<256xi8> {
    %selected = vectorchain.select_dynamic %activation, %ker_idx : vector<1024xi8>, vector<256xi2>, vector<256xi8>
    return %selected : vector<256xi8>
  }
}
