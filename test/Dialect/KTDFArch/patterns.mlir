// RUN: dataflow-scheduler-dialects-opt "%s" -ktdfarch-apply-patterns | FileCheck "%s" --check-prefix=DEFAULT
// RUN: dataflow-scheduler-dialects-opt "%s" -ktdfarch-apply-patterns="groups=widen" | FileCheck "%s" --check-prefix=WIDEN
// RUN: dataflow-scheduler-dialects-opt "%s" -ktdfarch-apply-patterns="groups=widen,map" | FileCheck "%s" --check-prefix=MAP

ktdf_arch.device @device {
  patterns [] {
    pdl.pattern : benefit(1) {
      %i64 = type : i64
      %magic = attribute = 0xAAAABBBBCCCCDDDD
      %op = operation "arith.constant" {"value" = %magic} -> (%i64: !pdl.type)
      rewrite {
        %magic2 = attribute = 42
        %cst_op = operation "arith.constant" {"value" = %magic2} -> (%i64: !pdl.type)
        %cst = result 0 of %cst_op
        replace %op with (%cst: !pdl.value)
      } 
    }
  }

  patterns ["widen"] {
    pdl.pattern : benefit(1) {
      %f16 = type : f16
      %in = operand : %f16
      %op = operation "math.exp" (%in: !pdl.value) -> (%f16: !pdl.type)
      rewrite {
        %f32 = type : f32
        %widen_op = operation "arith.extf" (%in: !pdl.value) -> (%f32: !pdl.type)
        %widen = result 0 of %widen_op
        %exp_op = operation "math.exp" (%widen: !pdl.value) -> (%f32: !pdl.type)
        %exp = result 0 of %exp_op
        %narrow_op = operation "arith.truncf" (%exp: !pdl.value) -> (%f16: !pdl.type)
        %narrow = result 0 of %narrow_op
        replace %op with (%narrow: !pdl.value)
      }
    }
  }

  patterns ["map"] {
    pdl.pattern : benefit(1) {
      %ty = type
      %in = operand : %ty
      %op = operation "math.exp" (%in: !pdl.value) -> (%ty: !pdl.type)
      apply_native_constraint "ktdf_arch.mapped_to" (%op: !pdl.operation) : !pdl.attribute {isNegated = true}
      rewrite {
        %mapped = attribute = "mapped"
        apply_native_rewrite "ktdf_arch.map_to" (%op, %mapped: !pdl.operation, !pdl.attribute)
      }
    }
  }
}

func.func @test(%arg0: f16) -> (f16, i64) attributes {ktdf_arch.maps_to = @device} {
  %exp = math.exp %arg0 : f16
  %magic = arith.constant 0xAAAABBBBCCCCDDDD : i64
  return %exp, %magic : f16, i64
}

// DEFAULT-LABEL: func.func @test(
// DEFAULT-SAME: %[[ARG0:.+]]: f16
// DEFAULT-DAG: %[[EXP:.+]] = math.exp %[[ARG0]] : f16
// DEFAULT-DAG: %[[MAGIC:.+]] = arith.constant 42 : i64
// DEFAULT: return %[[EXP]], %[[MAGIC:.+]]

// WIDEN-LABEL: func.func @test(
// WIDEN-SAME: %[[ARG0:.+]]: f16
// WIDEN-DAG: %[[WIDE:.+]] = arith.extf %[[ARG0]] : f16 to f32
// WIDEN-DAG: %[[EXP:.+]] = math.exp %[[WIDE]] : f32
// WIDEN-DAG: %[[NARROW:.+]] = arith.truncf %[[EXP]] : f32 to f16
// WIDEN-DAG: %[[MAGIC:.+]] = arith.constant 42 : i64
// WIDEN: return %[[NARROW]], %[[MAGIC:.+]]

// MAP-LABEL: func.func @test(
// MAP-SAME: %[[ARG0:.+]]: f16
// MAP-DAG: %[[WIDE:.+]] = arith.extf %[[ARG0]]  : f16 to f32
// MAP-DAG: %[[EXP:.+]] = math.exp %[[WIDE]] {ktdf_arch.maps_to = "mapped"} : f32
// MAP-DAG: %[[NARROW:.+]] = arith.truncf %[[EXP]] : f32 to f16
// MAP-DAG: %[[MAGIC:.+]] = arith.constant 42 : i64
// MAP: return %[[NARROW]], %[[MAGIC:.+]]
