// RUN:  dataflow-scheduler-dialects-opt %s | dataflow-scheduler-dialects-opt | FileCheck %s

// CHECK-LABEL:   func.func @test1() {
// CHECK-NEXT:     %[[VAL_0:.*]] = symbol.create_symbol : index
// CHECK-NEXT:     %[[VAL_1:.*]] = symbol.create_symbol : index
// CHECK-NEXT:     %[[VAL_2:.*]] = symbol.create_symbol : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-CL0-L3-SU", type = "l3su"} : index
// CHECK-NEXT:     dataflow.program_unit %[[VAL_6]] : {
// CHECK-NEXT:       affine.for %[[VAL_7:.*]] = 0 to %[[VAL_5]] {
// CHECK-NEXT:         %[[VAL_8:.*]] = symbol.symbol_immutable_mapping ({{\[}}%[[VAL_3]] -> %[[VAL_0]]], {{\[}}%[[VAL_4]] -> %[[VAL_1]]], {{\[}}%[[VAL_5]] -> %[[VAL_2]]]) : index
// CHECK-NEXT:         %[[VAL_9:.*]] = symbol.query_map(map:%[[VAL_8]], key:%[[VAL_7]]) : index
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.call @test1() : () -> ()

// Round-tripping test for create_symbol, symbol_immutable_mapping and query_map in symbol dialect.

module  {
  func.func @test1() {
    %symbol0 = symbol.create_symbol : index
    %symbol1 = symbol.create_symbol : index
    %symbol2 = symbol.create_symbol : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %l3su_c0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-CL0-L3-SU", type = "l3su"} : index
    dataflow.program_unit %l3su_c0 :  {
      affine.for %i = 0 to %c2 {
        %0 = symbol.symbol_immutable_mapping ([%c0 -> %symbol0],[%c1 -> %symbol1], [%c2 -> %symbol2]) : index
        %1 = symbol.query_map (map: %0, key: %i) : index
      }
    }
    return
  }
  func.call @test1() : () -> ()
}
