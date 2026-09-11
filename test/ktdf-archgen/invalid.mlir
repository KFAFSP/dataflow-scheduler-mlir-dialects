// RUN: not ktdf-archgen %s
// RUN: ktdf-archgen %s --verify-diagnostics

ktdf_arch.device @my_device {
    // expected-note@+1 {{previous definition is here}}
    exec_unit @collision
    // expected-error@+1 {{resource with id "collision" redefined}}
    exec_unit @collision
}
