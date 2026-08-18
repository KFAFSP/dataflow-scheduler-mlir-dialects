//===-- KTDFArchDialect.td ---------------------------------*- tablegen -*-===//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ktdf_arch dialect for describing device architectures.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_TD

include "mlir/IR/OpBase.td"

include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.td"

def KTDFArch_Dialect : Dialect {
  let name = "ktdf_arch";
  let cppNamespace = "::mlir::ktdf_arch";
  let summary = "Kernel Tile Dataflow (KTDF) Architecture Dialect";
  let description = [{
    The `ktdf_arch` dialect implements a language for modelling dataflow devices
    and their architecture suitable for scheduling dataflow applications. The
    representation is based on a graph model of memory and execution unit
    resources, which are connected by links.

    ## Intrinsics

    All *Resources* accept arbitrary sets of MLIR attributes. This allows users 
    to attach any payload they want. Intrinsics of the `ktdf_arch` dialect give
    some semantics to a well-known set of these attributes.

    Information carried in attributes is divided into two types: *Properties*
    which attach data to existing operations, and *Features* which describe sets
    of constraints satisfied or required by operations.

    ### Properties

    This dialect reserves its dialect attribute namespace (attribute names 
    prefixed by `ktdf_arch.*`) for *intrinsic properties* known to all users.

    |                   Name | Type           | Valid on       | Description
    | ---------------------: | :------------- | -------------- | --- 
    |            `bandwidth` | `i64` > 0      | any *Link*     | Maximum concurrent transfer bandwidth in bytes.
    |             `features` | `{Attribute}`  | any *Resource* or *Mappable* | Conjunction of *Features* (see below) provided or required.
    |               `maps_to`| `Attribute`    | any *Mappable* | Mapping constraint to a particular kind or unique ID.
    |             `overlaps` | `[Attribute]`  | any *Resource* | Two *Resources* with intersecting `overlaps` properties can't be mapped at the same time.
    | `transfer_granularity` | `i64` > 0      | any *Link*     | Allowed transfer sizes in bytes.

    ### Features

    *Resources* (and *Mappables*) can declare a set of *Features*, which define 
    the provided (or required) capabilities to be considered during mapping. A 
    *Feature* is an attribute, with the list of all *Features* on a *Resource* 
    (or *Mappable*) being considered a _conjunction_.

    The `ktdf_arch.feature.*` attribute name prefix is reserved by the 
    `ktdf_arch` dialect for features with pre-defined semantics. These include:

    |      Name | Valid on    | Meaning
    | --------: | ----------- | ---
    | `compute` | `exec_unit` | Can implement general-purpose computation.
    |    `load` | `exec_unit` | Can read from connected memory.
    |    `simd` | `exec_unit` | Has parallel SIMD lanes.
    |   `store` | `exec_unit` | Can write to connected memory.
    |   `queue` | `datapath`  | Can have multiple transactions in-flight.

    Additional features may be implemented in downstream dialects by 
    implementing the `ktdf_arch::FeatureDialectInterface`, which delegates
    verification and testing (requirement matching) to the dialect that owns
    the attribute namespace.

    #### `ktdf_arch.feature.load`/`ktdf_arch.feature.store`

    This feature is a record of the following fields:

    |                 Name | Type                    | Description
    | -------------------: | :---------------------- | ----
    |          `word_size` | `Attribute -> i64 > 0`? | Map from memory space to addressed data word size in bytes.
    | `access_granularity` | `Attribute -> Access`?  | Map from memory space to possible accesses.

    An absent `word_size` is treated as if it was `1`.

    An `Access` is a record of the following fields:

    |             Name | Type       | Description
    | ---------------: | :--------- | ---
    |  `size_in_words` | `i64 > 0`  | Maximum transfer size in words.
    | `align_in_words` | `i64 > 0`? | Minimum alignment requirement in words.

    An absent `align_in_words` is treated as if it was `1`.

    #### `ktdf_arch.feature.simd`

    This feature is a record of the following fields:

    |       Name | Type               | Description
    | ---------: | :----------------- | ---
    |    `splat` | `unit`?            | Whether the vector unit can splat scalars.
    | `zero_pad` | `unit`?            | Whether the vector unit can pad with zeros.
    |    `lanes` | `Type -> i64 > 0`? | Map from scalar type to number of vector lanes.

    #### `ktdf_arch.feature.queue`

    This feature is a record of the following fields:

    |      Name | Type       | Description
    | --------: | :--------- | ---
    | `ordered` | `unit`?    | Whether transfers remain ordered (FIFO).
    |    `size` | `i64 > 0`? | Maximum combined size of in-flight transfers in bytes.
    |   `depth` | `i64 > 0`? | Maximum number of in-flight transfers.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let hasOperationAttrVerify = 1;

  // NOTE: Although mlir-tablegen supports the `discardableAttrs` field, the
  //       helpers generated by this are not actually that helpful to us, since
  //       they don't help with verification and feature testing. Instead, we
  //       use custom fancy-ptr wrappers defined in `KTDFArchIntrinsics.h`.
  //       Unfortunately, we also can't use TableGen to inject additional
  //       definitions into the dialect class, so we must declare all the
  //       methods on the dialect instead of delegating to the wrapper classes
  //       directly.

  defvar intrinsicProperties = [
    KTDFArch_IntrinsicProperty<"bandwidth">,
    KTDFArch_IntrinsicProperty<"features">,
    KTDFArch_IntrinsicProperty<"maps_to">,
    KTDFArch_IntrinsicProperty<"overlaps">,
    KTDFArch_IntrinsicProperty<"transfer_granularity">
  ];

  defvar intrinsicFeatures = [
    KTDFArch_IntrinsicFeature<"compute">,
    KTDFArch_IntrinsicFeature<"load">,
    KTDFArch_IntrinsicFeature<"simd", "SIMD">,
    KTDFArch_IntrinsicFeature<"store">,
    KTDFArch_IntrinsicFeature<"queue">
  ];

  let extraClassDeclaration = [{
   private:
    void registerAttributes();
    void registerOps();
    void registerTypes();

  }] # KTDFArch_IntrinsicCodeGen<intrinsicProperties # intrinsicFeatures>.ret;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHDIALECT_TD