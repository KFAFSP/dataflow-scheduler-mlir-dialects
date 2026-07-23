ktdf_arch.device @my_device attributes {version = 1} {
  group share() {
    exec_unit {
      kind = "e0",
      ktdf_arch.features = { ktdf_arch.feature.compute }
    }
  }

  group { kind = "g0" } share() {
    group share() {
      exec_unit @special {
        kind = "e0",
        ktdf_arch.features = { ktdf_arch.feature.compute }
      }
    }

    group { kind = "g1" } share() {
      exec_unit {
        kind = "e1",
        ktdf_arch.features = { ktdf_arch.feature.compute }
      }
    }

    memory { kind = "m0", test.attr = 4 }
  }

  group { kind = "g1" } share() {
    exec_unit {
      kind = "e1",
      ktdf_arch.features = { ktdf_arch.feature.compute }
    }
  }

  group { kind = "g2" } share() {
    %a = exec_unit {
      kind = "e2",
      ktdf_arch.features = { ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 4> } }
    }

    %b = exec_unit {
      kind = "e3",
      ktdf_arch.features = { ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 8> } }
    }

    datapath { kind = "l0", ktdf_arch.bandwidth = 128 } %a to %b : exec_unit, exec_unit
  }
}
