%t0 = ktdf.create_token : !ktdf.token
%t1 = ktdf.create_token : !ktdf.token
%t2 = ktdf.create_token : !ktdf.token
%t3 = ktdf.create_token : !ktdf.token
%t4 = ktdf.create_token : !ktdf.token
%t5 = ktdf.create_token : !ktdf.token
%t6 = ktdf.create_token : !ktdf.token
%t7 = ktdf.create_token : !ktdf.token
"test.user"(%t0, %t1, %t2, %t3) : (!ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) -> ()

ktdf.pipeline {
  ktdf.stage depends_in(none) depends_out(%t0, %t1, %t2) {} {
    id = "a",
    expect_scc = ["a"],
    expect_consumers = ["c", "d", "d"],
    expect_producers = [],
    expect_depends_on = []
  }
  ktdf.stage depends_in(none) depends_out(%t1) {} {
    id = "b",
    expect_scc = ["b"],
    expect_consumers = ["c", "d"],
    expect_producers = [],
    expect_depends_on = []
  }
  ktdf.stage depends_in(%t1) depends_out(%t3) {} {
    id = "c",
    expect_scc = ["c"],
    expect_consumers = ["f"],
    expect_producers = ["a", "b"],
    expect_depends_on = ["a", "b"]
  }
  ktdf.stage depends_in(%t1, %t2) depends_out(none) {} {
    id = "d",
    expect_scc = ["d"],
    expect_consumers = [],
    expect_producers = ["a", "a", "b"],
    expect_depends_on = ["a", "b"]
  }
  ktdf.stage depends_in(none) depends_out(none) {} {
    id = "e",
    expect_scc = ["e"],
    expect_consumers = [],
    expect_producers = [],
    expect_depends_on = []
  }
  ktdf.stage depends_in(%t3) depends_out(none) {} {
    id = "f",
    expect_scc = ["f"],
    expect_consumers = [],
    expect_producers = ["c"],
    expect_depends_on = ["a", "b", "c"]
  }

  ktdf.stage depends_in(none) depends_out(%t4) {} {
    id = "j",
    expect_scc = ["j"],
    expect_consumers = ["g"],
    expect_producers = [],
    expect_depends_on = []
  }
  ktdf.stage depends_in(%t4, %t6) depends_out(%t5) {} {
    id = "g",
    expect_scc = ["g", "h"],
    expect_consumers = ["h"],
    expect_producers = ["j", "h"],
    expect_depends_on = ["h", "j"]
  }
  ktdf.stage depends_in(%t5) depends_out(%t6, %t7) {} {
    id = "h",
    expect_scc = ["g", "h"],
    expect_consumers = ["g", "i"],
    expect_producers = ["g"],
    expect_depends_on = ["g", "j"]
  }
  ktdf.stage depends_in(%t7) depends_out(none) {} {
    id = "i",
    expect_scc = ["i"],
    expect_consumers = [],
    expect_producers = ["h"],
    expect_depends_on = ["g", "h", "j"]
  }
}
