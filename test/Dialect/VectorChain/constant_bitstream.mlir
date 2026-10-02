// RUN: dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// Every value fits the element type, up to all of its bits. The values print
// as the bit patterns they are.

// CHECK-LABEL: func.func @constant_bitstream_i8
// CHECK: vectorchain.constant_bitstream {value = [0x0, 0x7f, 0xff]} : vector<1x3xi8>
func.func @constant_bitstream_i8() -> vector<1x3xi8> {
  %0 = vectorchain.constant_bitstream {value = [0, 127, 255]} : vector<1x3xi8>
  return %0 : vector<1x3xi8>
}

// CHECK-LABEL: func.func @constant_bitstream_i64
// CHECK: vectorchain.constant_bitstream {value = [0x7f, 0x81, 0xffffffffffffffff]} : vector<1x3xi64>
func.func @constant_bitstream_i64() -> vector<1x3xi64> {
  %0 = vectorchain.constant_bitstream {value = [127, 129, -1]} : vector<1x3xi64>
  return %0 : vector<1x3xi64>
}

// CHECK-LABEL: func.func @constant_bitstream_f64
// CHECK: vectorchain.constant_bitstream {value = [0x3ff0000000000000, 0x0]} : vector<1x2xf64>
func.func @constant_bitstream_f64() -> vector<1x2xf64> {
  %0 = vectorchain.constant_bitstream {value = [4607182418800017408, 0]} : vector<1x2xf64>
  return %0 : vector<1x2xf64>
}
