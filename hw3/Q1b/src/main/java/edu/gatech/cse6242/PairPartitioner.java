package edu.gatech.cse6242;


import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Partitioner;

public class PairPartitioner extends Partitioner<GraphPair,NullWritable>{
    @Override
    public int getPartition(GraphPair pair, NullWritable nullWritable,int numPartitions){
        return pair.getReceiver().hashCode() % numPartitions;
    }
}