// RUN: dataflow-scheduler-dialects-opt --ktdfarch-instantiate-neighborhoods %s --verify-diagnostics --split-input-file

ktdf_arch.device @degenerate {
  // expected-error@+1 {{unable to instantiate neighborhood}}
  %nb = neighborhood %self : (exec_unit, exec_unit)[1] {
    %a = exec_unit @a
    %b = exec_unit @b
    yield %a, %b
  }
  // expected-error@+1 {{unable to resolve neighbor}}
  %x, %y = neighbor affine_map<() -> (2)> in %nb : (exec_unit, exec_unit)[1]

  datapath %x to %y : exec_unit, exec_unit
}

// -----

ktdf_arch.device @out_of_bounds {
  // expected-error@+1 {{unable to instantiate neighborhood}}
  %outer = neighborhood %self_o : (exec_unit)[2] {
    %inner = neighborhood %self_i : (exec_unit)[3] {
      %e = exec_unit @exec
      yield %e
    }
    %e0 = neighbor affine_map<(d0) -> (d0+1)> in %inner : (exec_unit)[3]
    yield %e0
  }
  %a = neighbor affine_map<() -> (0)> in %outer : (exec_unit)[2]
  // expected-error@+1 {{unable to resolve neighbor}}
  %b = neighbor affine_map<() -> (2)> in %outer : (exec_unit)[2]

  datapath %a to %b : exec_unit, exec_unit
}
