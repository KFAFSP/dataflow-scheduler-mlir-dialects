// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

ktdf_arch.device @memory_kind_unset {
  // expected-error@+1 {{attribute 'kind' requires valid memory space}}
  memory
}

// -----

ktdf_arch.device @memory_kind_invalid_space {
  // expected-error@+1 {{attribute 'kind' requires valid memory space}}
  memory { kind = 1.23 }
}

// -----

ktdf_arch.device @bandwidth_not_on_link {
  // expected-error@+1 {{only valid on Link}}
  exec_unit { ktdf_arch.bandwidth = 1 }
}

// -----

ktdf_arch.device @bandwidth_not_an_integer {
  %a = exec_unit
  %b = exec_unit
  // expected-error@+1 {{requires 64-bit integer attribute}}
  datapath { ktdf_arch.bandwidth = "a" } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @bandwidth_less_than_one {
  %a = exec_unit
  %b = exec_unit
  // expected-error@+1 {{value must be > 0}}
  datapath { ktdf_arch.bandwidth = 0 } %a to %b : exec_unit, exec_unit
}

// -----

// expected-error@+1 {{requires dictionary attribute}}
func.func private @features_not_a_dict() attributes { ktdf_arch.features = "a" }

// -----

ktdf_arch.device @maps_to_not_mappable {
  // expected-error@+1 {{only valid on mappable ops}}
  exec_unit { ktdf_arch.maps_to = 4 }
}

// -----

ktdf_arch.device @overlaps_not_array {
  // expected-error@+1 {{requires array attribute}}
  exec_unit { ktdf_arch.overlaps = 4 }
}

// -----

ktdf_arch.device @size_not_an_integer {
  // expected-error@+1 {{64-bit integer attribute}}
  memory { kind = "A", size = 1.4 }
}

// -----

ktdf_arch.device @size_less_than_one {
  // expected-error@+1 {{minimum value is 1}}
  memory { kind = "A", size = 0 }
}

// -----

ktdf_arch.device @transfer_granularity_not_on_link {
  // expected-error@+1 {{only valid on Link}}
  exec_unit { ktdf_arch.transfer_granularity = array<i64: 0> }
}

// -----

ktdf_arch.device @transfer_granularity_not_i64_array {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{requires dense 64-bit integer array attribute}}
  datapath { ktdf_arch.transfer_granularity = [0, 1] } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @transfer_granularity_repeated {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{repeated granularity (1)}}
  datapath { ktdf_arch.transfer_granularity = array<i64: 1, 1> } %a to %b : exec_unit, exec_unit
}
