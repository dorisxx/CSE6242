hadoop jar ./target/q1a-1.0.jar edu.gatech.cse6242.Q1a /user/cse6242/graph1.tsv /user/cse6242/q1aoutput1
hadoop fs -getmerge /user/cse6242/q1aoutput1/ q1aoutput1.tsv
hadoop fs -rm -r /user/cse6242/q1aoutput1
