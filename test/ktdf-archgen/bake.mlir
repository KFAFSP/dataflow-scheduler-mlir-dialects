// RUN: ktdf-archgen %s -o - | FileCheck %s

// CHECK-LABEL: ktdf_arch.device @ring
ktdf_arch.device @ring {
  // CHECK: %[[EXEC0:.+]] = group #ring_element share()
  // CHECK: %[[SW0:.+]]:3 = switch[3]
  // CHECK-DAG: datapath %[[SW0]]#2 to %[[EXEC0]] : port, exec_unit
  // CHECK-DAG: datapath %[[EXEC0]] to %[[SW0]]#2 : exec_unit, port
  // CHECK: datapath %[[SW2:.+]]#1 to %[[SW0]]#0 : port, port

  // CHECK: %[[EXEC1:.+]] = group #ring_element share()
  // CHECK: %[[SW1:.+]]:3 = switch[3]
  // CHECK-DAG: datapath %[[SW1]]#2 to %[[EXEC1]] : port, exec_unit
  // CHECK-DAG: datapath %[[EXEC1]] to %[[SW1]]#2 : exec_unit, port
  // CHECK: datapath %[[SW0]]#1 to %[[SW1]]#0 : port, port

  // CHECK: %[[EXEC2:.+]] = group #ring_element share()
  // CHECK: %[[SW2]]:3 = switch[3]
  // CHECK-DAG: datapath %[[SW2]]#2 to %[[EXEC2]] : port, exec_unit
  // CHECK-DAG: datapath %[[EXEC2]] to %[[SW2]]#2 : exec_unit, port
  // CHECK: datapath %[[SW1]]#1 to %[[SW2]]#0 : port, port

  %ring = neighborhood %self : (port, exec_unit)[3] {
    %exec = group {kind="ring_element"} share() {
      %local = memory {kind = "local"}
      %exec = exec_unit @exec
      yield %exec
    } -> exec_unit

    %sw:3 = switch[3]
    datapath %sw#2 to %exec : port, exec_unit
    datapath %exec to %sw#2 : exec_unit, port

    %prev_out, %prev_exec = neighbor affine_map<(d0) -> ((d0 - 1) mod 3)> in %self : (port, exec_unit)[3]
    datapath %prev_out to %sw#0 : port, port

    yield %sw#1, %exec : port, exec_unit
  }

  %out_0, %exec_0 = neighbor affine_map<() -> (0)> in %ring : (port, exec_unit)[3]
  %out_2, %exec_2 = neighbor affine_map<() -> (2)> in %ring : (port, exec_unit)[3]

  // CHECK: datapath #skip_one %[[EXEC0]] to %[[EXEC2]]
  datapath {kind = "skip_one"} %exec_0 to %exec_2 : exec_unit, exec_unit
}
