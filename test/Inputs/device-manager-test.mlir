ktdf_arch.device @inline_device {
  exec_unit
}

ktdf_arch.device @missing_device import("DO_NOT_CREATE_A_FILE_WITH_THIS_NAME")

ktdf_arch.device @device attributes {version = 2, overridable = 2} import("device.mlir")

ktdf_arch.device @invalid_device import("device-invalid.mlir")

ktdf_arch.device @recursive_device import("device-manager-test.mlir")

func.func private @test_1()

func.func private @test_2()
