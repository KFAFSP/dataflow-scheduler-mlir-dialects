// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// expected-error@+1 {{symbol name}}
ktdf_arch.device {}

// -----

// expected-error@+1 {{empty device requires 'import_path' attribute}}
"ktdf_arch.device"() ({}) {sym_name = "empty_no_import"} : () -> ()

// -----

// expected-error@+1 {{non-empty device can't have 'import_path' attribute}}
ktdf_arch.device @non_empty_with_import attributes { import_path = "" } {
  exec_unit
}

// -----

ktdf_arch.device @sibling_redefines_id {
  // expected-note@+1 {{previous definition is here}}
  exec_unit @a
  // expected-error@+1 {{resource with id "a" redefined}}
  exec_unit @a
}

// -----

ktdf_arch.device @cousin_redefines_id {
  group share() {
    // expected-note@+1 {{previous definition is here}}
    exec_unit @a
  }
  // expected-error@+1 {{resource with id "a" redefined}}
  exec_unit @a
}

// -----

// expected-note@+1 {{previous definition is here}}
ktdf_arch.device @id_is_device_name {
  // expected-error@+1 {{resource with id "id_is_device_name" redefined}}
  exec_unit @id_is_device_name
}

// -----

// expected-error@+1 {{expects child ops to be resources}}
ktdf_arch.device @unexpected_op {
  // expected-note@+1 {{unexpected child is here}}
  "dialect.op"() : () -> ()
}
