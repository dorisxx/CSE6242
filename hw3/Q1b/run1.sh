hadoop jar ./target/q1b-1.0.jar edu.gatech.cse6242.Q1b /user/cse6242/graph1.tsv /user/cse6242/q1boutput1
hadoop fs -getmerge /user/cse6242/q1boutput1/ q1boutput1.tsv
hadoop fs -rm -r /user/cse6242/q1boutput1
