// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

// One bit more than the element type holds.
func.func @constant_bitstream_i8_overflow() -> vector<1x1xi8> {
  // expected-error @+1 {{value attribute contains element that exceeds bitwidth of output element type}}
  %0 = vectorchain.constant_bitstream {value = [256]} : vector<1x1xi8>
  return %0 : vector<1x1xi8>
}

// -----

// A negative value sets bits above the element type's.
func.func @constant_bitstream_i32_negative() -> vector<1x1xi32> {
  // expected-error @+1 {{value attribute contains element that exceeds bitwidth of output element type}}
  %0 = vectorchain.constant_bitstream {value = [-1]} : vector<1x1xi32>
  return %0 : vector<1x1xi32>
}
