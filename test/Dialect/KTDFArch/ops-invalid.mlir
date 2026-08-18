// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

ktdf_arch.device @patterns_invalid_child {
  // expected-error@+1 {{expects child ops to be 'pdl.pattern'}}
  patterns {
    // expected-note@+1 {{unexpected child is here}}
    "dialect.op" () : () -> ()
  }
}
