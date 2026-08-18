// RUN: dataflow-scheduler-dialects-opt --allow-unregistered-dialect | dataflow-scheduler-dialects-opt

ktdf_arch.device @my_device attributes {version = 1} {
  %ddr = memory {
    kind = "DDR",
    size = 17179869184
  }

  %cpu = exec_unit { 
    kind = "CPU",
    ktdf_arch.features = { 
      ktdf_arch.feature.compute, 
      ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f32 = 4, f16 = 8> } 
    }
  }
  %ls = exec_unit { 
    ktdf_arch.features = {
      ktdf_arch.feature.load = {
        word_size = #ktdf_arch.map<"DDR" = 1>,
        access_granularity = #ktdf_arch.map<
          "DDR" = [
            {size_in_words = 1, align_in_words = 1},
            {size_in_words = 128, align_in_words = 128}
          ]
        >
      },
      ktdf_arch.feature.store = {
        word_size = #ktdf_arch.map<"DDR" = 1>,
        access_granularity = #ktdf_arch.map<
          "DDR" = [
            {size_in_words = 1, align_in_words = 1},
            {size_in_words = 128, align_in_words = 128}
          ]
        >
      },
      ktdf_arch.feature.simd = { splat, zero_pad }
    }
  }

  datapath { 
    ktdf_arch.bandwidth = 128, 
    ktdf_arch.transfer_granularity = array<i64: 4, 8, 16> 
  } %ddr to %ls : memory, exec_unit
  datapath { 
    ktdf_arch.overlaps = ["LS_queue"], 
    ktdf_arch.features = { 
      ktdf_arch.feature.queue = { size = 4096, depth = 256, ordered } 
    } 
  } %ls to %cpu : exec_unit, exec_unit 
  datapath { ktdf_arch.overlaps = ["LS_queue"] } %cpu to %ls : exec_unit, exec_unit
  datapath %ls to %ddr : exec_unit, memory
}

func.func @my_func(%arg0: tensor<4xf32>) attributes { ktdf_arch.maps_to = @my_device } {
  arith.addf %arg0, %arg0 { 
    ktdf_arch.maps_to = "CPU", 
    ktdf_arch.features = { ktdf_arch.feature.simd } 
  } : tensor<4xf32>

  return
}