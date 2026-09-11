// RUN: ktdf-archgen %s -o - | FileCheck %s

// CHECK-DAG: #exec = {kind = "exec"}
// CHECK-DAG: #mem = {kind = "mem"}
// CHECK-DAG: #switch = {kind = "switch"}
// CHECK-DAG: #group = {kind = "group"}

// CHECK-LABEL: @my_device
ktdf_arch.device @my_device {
  // CHECK-NEXT: exec_unit @exec #exec
  exec_unit @exec {kind = "exec"}
  // CHECK-NEXT: memory @mem #mem
  memory @mem {kind = "mem"}
  // CHECK-NEXT: switch[3] @switch #switch
  switch[3] @switch {kind = "switch"}
  // CHECK-NEXT: group @group #group share()
  group @group {kind = "group"} share() {}
}
