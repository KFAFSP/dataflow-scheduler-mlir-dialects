# `Dataflow Scheduler MLIR Dialects`

**Dataflow Scheduler MLIR Dialects** is a set of MLIR dialect definitions for
scheduling tiled compute kernels onto dataflow hardware. Its core dialect,
`ktdf`, models a program as a pipeline of stages that execute concurrently and
synchronize through explicit dataflow tokens, making the overlap of computation
and data movement directly expressible in the IR. Data transfers between memory
spaces and FIFO channels are first-class operations, and the dialect provides
supporting abstractions for loop tiling, double buffering, and distributing
parallel iteration spaces across hardware instances. Together these let a
scheduler reason about — and a backend lower — the concurrency and memory
traffic of a kernel on dataflow accelerators.

## Description

This project provides the dialect definitions for dataflow-based schedulers. Currently, these include:

|           Name | Description                                                                                                                           |
| -------------: |:--------------------------------------------------------------------------------------------------------------------------------------|
| `ktdf`         | The Kernel Tile Dataflow (KTDF) dialect, which adds a dataflow pipeline abstraction suitable for scheduling to dataflow accelerators. |
| `ktdf_arch`    | Models dataflow devices and their architecture as a graph of memory and execution-unit resources connected by links, for scheduling dataflow applications. |
| `ktdf_lowering`| Temporary operations used during the multi-phase lowering from KTIR (the `ktdf` dialect) to Dataflow IR. |
| `dataflow`     | The Dataflow IR dialect: program units, inter-unit communication (`send`/`receive`), explicit synchronization, and logical memory views. |
| `agen`         | Address-generation memory operations — vector loads/stores and composite load-and-store transfers driven by affine maps and sets. |
| `vectorchain`  | Advanced vector arithmetic — element-wise binary ops with reduction, and shuffle/selection of vector elements. |
| `uniform`      | Uniformization operations that capture constant differences across per-core programs of a unit (immutable mappings and queries). |

## Prerequisites

Before building `DataflowSchedulerDialects`, you need the following dependencies:

1. **Build Tools**
   - Git
   - CMake >= 3.21.0
   - C++17 compiler (clang >= 21.0.0 recommended)
   - Ninja build (optional, recommended)
   - Python >= 3.12 (optional)

2. **LLVM Project** - The LLVM/MLIR infrastructure
   
   You can build this project using a local installation of LLVM & MLIR. However, this project requires the specific revision `llvmorg-22.1.3` (SHA: `e9846648`).

   If you do not have such an installation, you can build it using:
  
   ```sh
   # Clone the repository as lean as possible.
   git clone https://github.com/llvm/llvm-project.git \
     --branch llvmorg-22.1.3 --sparse --depth 1
   cd llvm-project
   git sparse-checkout add cmake libc llvm mlir runtimes third-party

   # Configure LLVM & MLIR.
   cmake -S ./llvm -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_PROJECTS=mlir \
     -DLLVM_TARGETS_TO_BUILD="host"
   # If you want to build the Python bindings, add:
     # -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
     # -DMLIR_PYTHON_STUBGEN_ENABLED=ON
   # If you are a developer, add:
     # -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     # -DBUILD_SHARED_LIBS=ON \
     # -DLLVM_ENABLE_ASSERTIONS=ON

   # Build LLVM. This may take a while, and consume a lot of space.
   cmake --build build
   ```

## Build Instructions

Building `DataflowSchedulerDialects` requires a working build or installation
of LLVM and MLIR to be present on your system.

```sh
cd scheduler-mlir-dialects

# Configure the project.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$MLIR_INSTALL_PREFIX/lib/cmake/mlir

# Build everything.
cmake --build build --target all
```

The following CMake variables can be configured:

|              Name | Type      | Description |
| ----------------: | :-------- | --- |
| `MLIR_DIR` <br/>*(required)* | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-22.1.3/lib/cmake/mlir`* |
| `LLVM_EXTERNAL_LIT` <br/>*(optional)* | `STRING` | Path to a `lit` executable, required for testing. |
| `DataflowSchedulerDialects_ENABLE_PYTHON_BINDINGS` <br/>*(default: `OFF`)* | `BOOL` | Whether to build the Python bindings. |
| `DataflowSchedulerDialects_BUILD_TOOLS` <br/>*(default: `ON`)* | `BOOL` | Whether to build the tool executables. |
| `DataflowSchedulerDialects_ENABLE_TESTS` <br/>*(default: `ON`)* | `BOOL` | Whether to build and run the unit tests. |
| `MLIR_LINK_MLIR_DYLIB` <br/>*(optional)* | `BOOL` | Whether to link against a shared MLIR library. |

### **Linking against shared LLVM/MLIR**

This project supports linking against a version of LLVM/MLIR that was built using the `LLVM_BUILD_LLVM_DYLIB` etc. flags. Since MLIR does not export the `MLIR_LINK_MLIR_DYLIB` flag, it is inferred from the presence of the `MLIR` library target when it is not specified at configure time. When linking against shared MLIR, CMake will set up an RPATH to your MLIR install destination.

## Documentation

In addition to the documentation found in the project sources, there are auto-generated documentations for the MLIR dialects, interfaces and passes. These can be generated to the build directory using:

```sh
cmake --build build --target dataflow-scheduler-dialects-doc
```

## Running Tests

The regression tests can be run with the following command:

```sh
cmake --build build --target check-dataflow-scheduler-dialects
```

## Installing

You can install the generated package to `$INSTALL_PREFIX` using the command:

```sh
cmake --install build --prefix $INSTALL_PREFIX
```

Note that the contents of your install will depend on the targets you have built so far. In particular, building the documentation target before installing will cause the docs to be installed. You can also select specific components for installation.
