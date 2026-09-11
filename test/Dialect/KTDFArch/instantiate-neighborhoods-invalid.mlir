// RUN: dataflow-scheduler-dialects-opt --ktdfarch-instantiate-neighborhoods %s --verify-diagnostics

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
