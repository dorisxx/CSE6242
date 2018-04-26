hadoop jar ./target/q1b-1.0.jar edu.gatech.cse6242.Q1b /user/cse6242/graph2.tsv /user/cse6242/q1boutput2
hadoop fs -getmerge /user/cse6242/q1boutput2/ q1boutput2.tsv
hadoop fs -rm -r /user/cse6242/q1boutput2
