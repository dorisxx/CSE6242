hadoop jar ./target/q1a-1.0.jar edu.gatech.cse6242.Q1a /user/cse6242/graph2.tsv /user/cse6242/q1aoutput2
hadoop fs -getmerge /user/cse6242/q1aoutput2/ q1aoutput2.tsv
hadoop fs -rm -r /user/cse6242/q1aoutput2