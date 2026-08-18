// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect --split-input-file %s | dataflow-scheduler-dialects-opt --allow-unregistered-dialect

ktdf_arch.device @my_device {
  // Define one memory and one compute resource.
  %main_memory = memory { kind = 1 }
  %main_exec = exec_unit

  // Establish a bi-directional link between the two.
  datapath %main_memory to %main_exec : memory, exec_unit
  datapath %main_exec to %main_memory : exec_unit, memory
}

// Import an architecture graph that is read from disk later.
ktdf_arch.device @my_imported_device import("./relative/to/input.mlir")

// -----

ktdf_arch.device @my_device {
  %shared_a = memory { kind = "shared_a" }
  %shared_b = memory { kind = "shared_b" }

  // The following 3 entities are locally similar, i.e., their internal
  // resources are the same. Furthermore:
  // - @grp0 and @grp1 are similar, but their list of captured 
  //   resources differs, meaning that they are _not_ identical.
  // - @grp0 and @grp2 are similar, and their list of captures
  //   resources is the same, meaning that they _must be_ identical.

  %c0 = group @grp0 { kind = "core" } share(%shared_a) {
    %private = memory { kind = "private" }
    datapath %shared_a to %private : memory, memory
    datapath %private to %shared_a : memory, memory

    %compute = exec_unit
    datapath %private to %compute : memory, exec_unit
    datapath %compute to %private : exec_unit, memory
    yield %compute
  } -> exec_unit
  group @grp1 { kind = "core" } share(%shared_b) {
    %private = memory { kind = "private" }
    datapath %shared_b to %private : memory, memory
    datapath %private to %shared_b : memory, memory

    %compute = exec_unit
    datapath %private to %compute : memory, exec_unit
    datapath %compute to %private : exec_unit, memory
  }
  %c2 = group @grp2 { kind = "core" } share(%shared_a) {
    %private = memory { kind = "private" }
    datapath %shared_a to %private : memory, memory
    datapath %private to %shared_a : memory, memory

    %compute = exec_unit
    datapath %private to %compute : memory, exec_unit
    datapath %compute to %private : exec_unit, memory
    yield %compute
  } -> exec_unit

  datapath %c0 to %c2 : exec_unit, exec_unit
  datapath %c2 to %c0 : exec_unit, exec_unit
}

// -----

ktdf_arch.device @my_device {
  %c0 = exec_unit
  
  %c1 = group share() {
    %c1 = exec_unit
    yield %c1
  } -> exec_unit

  datapath %c0 to %c1 : exec_unit, exec_unit
  datapath %c1 to %c0 : exec_unit, exec_unit
}

// -----

ktdf_arch.device @my_device {
  %ram = memory { kind = #ptr.generic_space }
  %ddr = memory { kind = "DDR", size = 17179869184 }
}

// -----

ktdf_arch.device @my_device {
  %sw0:2 = switch [2]
  %sw1:2 = switch [2] { connectivity = dense<[[0, 1],[1, 0]]> : tensor<2x2xi1> }
  %sw2:4 = switch [4] { connectivity = sparse<[[0, 2], [0, 3], [1, 2]], true> : tensor<4x4xi1> }
}

// -----

ktdf_arch.device @my_device {
  %m = memory { kind = 1 }
  %c = exec_unit
  %sw:2 = switch [2]

  datapath @c_dma %m to %c : memory, exec_unit
  datapath { kind = "fifo" } %c to %sw#0 : exec_unit, port
}

// -----

func.func private @map_attr() attributes { 
  map = #ktdf_arch.map<f32 = 1, { hello } = "world"> 
}
