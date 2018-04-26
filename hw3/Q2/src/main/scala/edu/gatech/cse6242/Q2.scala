package edu.gatech.cse6242

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object Q2 {

	def main(args: Array[String]) {
    	val sc = new SparkContext(new SparkConf().setAppName("Q2"))
		val sqlContext = new SQLContext(sc)
		import sqlContext.implicits._

    	// read the file
    	val file = sc.textFile("hdfs://localhost:8020" + args(0))
		/* TODO: Needs to be implemented */
		val filtered = file.map(line => line.split("\t")).filter(x => x.last.toInt >= 10)
		val outbound = filtered.map(line => (line(0),(line(2).toInt,1)))
				.reduceByKey{case((val1,count1),(val2,count2)) =>
								(val1+val2,count1+count2)}
				.mapValues{
					case(sum,count) => sum.toDouble/count.toDouble
				}
		val inbound = filtered.map(line => (line(1),(line(2).toInt,1)))
				.reduceByKey{case((val1,count1),(val2,count2)) =>
								(val1+val2,count1+count2)}
				.mapValues{
					case(sum,count) => sum.toDouble/count.toDouble
				}
		val bound = outbound.fullOuterJoin(inbound)
		val gross_weight = bound.mapValues{case(Some(a),Some(b)) => a-b
										   case(None,Some(b)) => -b
										   case(Some(a),None)=> a}
		val output = gross_weight.collect.map(x => x._1 +"\t"+x._2)

    	// store output on given HDFS path.
    	// YOU NEED TO CHANGE THIS
    	sc.makeRDD(output).saveAsTextFile("hdfs://localhost:8020" + args(1))
  	}
}
