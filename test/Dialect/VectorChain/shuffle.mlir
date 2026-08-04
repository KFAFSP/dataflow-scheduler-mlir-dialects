// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @shuffle_splat(
// CHECK-SAME:      %arg0: vector<1xf16>) -> vector<64xf16> {
// CHECK-NEXT:      %0 = vectorchain.shuffle input(%arg0) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK-NEXT:      return %0 : vector<64xf16>
// CHECK-NEXT:    }

module {
  func.func @shuffle_splat(%input: vector<1xf16>) -> vector<64xf16> {
    %result = vectorchain.shuffle input(%input) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
    return %result : vector<64xf16>
  }
}
