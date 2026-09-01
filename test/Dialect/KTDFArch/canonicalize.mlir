// RUN: dataflow-scheduler-dialects-opt --canonicalize %s | FileCheck %s

// CHECK-DAG: #[[MAP0:.+]] = affine_map<() -> (0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<() -> (1)>

// CHECK-LABEL: ktdf_arch.device @dead_neighborhood
ktdf_arch.device @dead_neighborhood {
  // CHECK: exec_unit @exec
  // CHECK-NOT: neighborhood
  %nb = neighborhood %self : ()[] {
    exec_unit @exec
    neighbor affine_map<() -> ()> in %self : ()[]
  }
  neighbor affine_map<() -> ()> in %nb : ()[]
}

// CHECK-LABEL: ktdf_arch.device @inline_singleton_neighborhood
ktdf_arch.device @inline_singleton_neighborhood {
  // CHECK-NOT: neighborhood
  // CHECK-DAG: %[[EXEC:.+]] = exec_unit @exec
  // CHECK-DAG: %[[MEM:.+]] = memory @mem
  %nb = neighborhood %self : ()[] {
    %exec = exec_unit @exec
    %mem = memory @mem {kind = "mem"}
    datapath %exec to %mem : exec_unit, memory
  }
}

// CHECK-LABEL: ktdf_arch.device @resolve_neighbor
ktdf_arch.device @resolve_neighbor {
  // CHECK-DAG: %[[ONE:.+]] = neighborhood {kind = "one"}
  %one = neighborhood {kind = "one"} %arg1 : (exec_unit)[2] {
    %exec = exec_unit
    yield %exec
  }
  // CHECK-DAG: %[[TWO:.+]] = neighborhood {kind = "two"}
  %two = neighborhood {kind = "two"} %arg1 : (exec_unit)[2] {
    %exec = exec_unit
    yield %exec
  }
  // CHECK: %[[EXE0:.+]] = neighbor #[[MAP1]] in %[[ONE]] :
  %exe0 = neighbor affine_map<() -> (0, 1)> in %one, %two : (exec_unit)[2]
  // CHECK: %[[EXE1:.+]] = neighbor #[[MAP0]] in %[[TWO]] :
  %exe1 = neighbor affine_map<() -> (1, 0)> in %one, %two : (exec_unit)[2]
  datapath %exe0 to %exe1 : exec_unit, exec_unit
}

// CHECK-LABEL: ktdf_arch.device @replace_neighbor
ktdf_arch.device @replace_neighbor {
  // CHECK-DAG: %[[EXEC:.+]] = exec_unit
  %exec = exec_unit
  %nb = neighborhood {kind = "one"} %arg1 : (exec_unit)[] {
    yield %exec
  }
  %exe = neighbor affine_map<() -> ()> in %nb : (exec_unit)[]
  // CHECK-DAG: %[[MEM:.+]] = memory
  %mem = memory {kind = "mem"}
  // CHECK: datapath %[[EXEC]] to %[[MEM]]
  datapath %exe to %mem : exec_unit, memory
}
