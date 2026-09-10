// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

ktdf_arch.device @memory_invalid_size {
  // expected-error@+1 {{64-bit integer attribute whose minimum value is 1}}
  memory {size = "A"}
}

ktdf_arch.device @switch_invalid_connectivity {
  // expected-error@+1 {{directed sparse or dense adjacency matrix}}
  switch[3] {connectivity = []}
}

ktdf_arch.device @patterns_invalid_child {
  // expected-error@+1 {{expects child ops to be 'pdl.pattern'}}
  patterns {
    // expected-note@+1 {{unexpected child is here}}
    "dialect.op" () : () -> ()
  }
}
