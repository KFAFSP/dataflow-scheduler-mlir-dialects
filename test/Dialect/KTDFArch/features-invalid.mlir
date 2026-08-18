// RUN: dataflow-scheduler-dialects-opt --split-input-file --verify-diagnostics %s

ktdf_arch.device @compute_not_on_execution_unit {
  // expected-error@+1 {{only valid on execution units}}
  memory { kind="A", ktdf_arch.features = { ktdf_arch.feature.compute } }
}

// -----

ktdf_arch.device @load_not_on_execution_unit {
  // expected-error@+1 {{only valid on execution units}}
  memory { kind="A", ktdf_arch.features = { ktdf_arch.feature.load } }
}

// -----

ktdf_arch.device @load_invalid_word_size_map {
  // expected-error@+1 {{'word_size' requires map}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { word_size = 1 } } }
}

// -----

ktdf_arch.device @load_invalid_word_size_invalid {
  // expected-error@+1 {{'word_size["A"]' must be >= 0}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { word_size = #ktdf_arch.map<"A" = -1> } } }
}

// -----

ktdf_arch.device @load_invalid_access_granularity_map {
  // expected-error@+1 {{'access_granularity' requires map}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { access_granularity = 1 } } }
}

// -----

ktdf_arch.device @load_invalid_size_type {
  // expected-error@+1 {{'access_granularity["A"]' attribute 'size_in_words' requires 64-bit integer}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { access_granularity = #ktdf_arch.map<"A" = [{size_in_words = 1 : i32}]> } } }
}

// -----

ktdf_arch.device @load_invalid_size {
  // expected-error@+1 {{'access_granularity["A"]' attribute 'size_in_words' must be >= 0}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { access_granularity = #ktdf_arch.map<"A" = [{size_in_words = -1}]> } } }
}

// -----

ktdf_arch.device @load_invalid_align {
  // expected-error@+1 {{'access_granularity["A"]' attribute 'align_in_words' must be >= 0}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.load = { access_granularity = #ktdf_arch.map<"A" = [{size_in_words = 1, align_in_words = 0}]> } } }
}

// -----

ktdf_arch.device @simd_not_on_execution_unit {
  // expected-error@+1 {{only valid on execution units}}
  memory { kind="A", ktdf_arch.features = { ktdf_arch.feature.simd } }
}

// -----

ktdf_arch.device @simd_splat_not_unit {
  // expected-error@+1 {{'splat' requires unit}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.simd = { splat = 1 } } }
}

// -----

ktdf_arch.device @simd_zero_pad_not_unit {
  // expected-error@+1 {{'zero_pad' requires unit}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.simd = { zero_pad = 1 } } }
}

// -----

ktdf_arch.device @simd_invalid_lanes {
  // expected-error@+1 {{'lanes' requires map from type to 64-bit integer}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.simd = { lanes = 1 } } }
}

// -----

ktdf_arch.device @queue_not_on_link {
  // expected-error@+1 {{only valid on links}}
  exec_unit { ktdf_arch.features = { ktdf_arch.feature.queue = { size = "a" } } }
}

// -----

ktdf_arch.device @queue_ordered_not_unit {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{'ordered' requires unit}}
  datapath { ktdf_arch.features = { ktdf_arch.feature.queue = { ordered = true } } } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @queue_size_not_int {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{'size' requires 64-bit integer}}
  datapath { ktdf_arch.features = { ktdf_arch.feature.queue = { size = "a" } } } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @queue_size_zero {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{'size' must be > 0}}
  datapath { ktdf_arch.features = { ktdf_arch.feature.queue = { size = 0 } } } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @queue_depth_not_int {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{'depth' requires 64-bit integer}}
  datapath { ktdf_arch.features = { ktdf_arch.feature.queue = { depth = "a" } } } %a to %b : exec_unit, exec_unit
}

// -----

ktdf_arch.device @queue_depth_zero {
  %a = exec_unit
  %b = exec_unit

  // expected-error@+1 {{'depth' must be > 0}}
  datapath { ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 0 } } } %a to %b : exec_unit, exec_unit
}
