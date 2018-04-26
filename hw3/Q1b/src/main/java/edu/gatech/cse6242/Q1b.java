package edu.gatech.cse6242;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Q1b {

	public static class TokenizerMapper
	extends Mapper<LongWritable,Text,GraphPair,NullWritable>{
		private GraphPair pair = new GraphPair();
		private NullWritable nullValue = NullWritable.get();
		
		@Override
		protected void map(LongWritable key, Text value, Context context
				)throws IOException, InterruptedException{
			String[] fields = value.toString().split("\\t");
			if(fields.length==3){
				pair.setReceiver(fields[2]);
				int w = Integer.parseInt(fields[1]);
				int sender = Integer.parseInt(fields[0]);

				pair.setWeight(w);
				pair.setSender(sender);
				context.write(pair,nullValue);
			}
		}
	}
	public static class SecondarySortingReducer extends Reducer<GraphPair,NullWritable, Text, IntWritable>{
		@Override
		protected void reduce(GraphPair key, Iterable<NullWritable> values, Context context
				)throws IOException,InterruptedException{
					context.write(key.getReceiver(),key.getSender());
				}
	}
	
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Q1b");

    /* TO DO: Needs to be implemented */
	
    FileInputFormat.addInputPath(job, new Path(args[0]));
	FileOutputFormat.setOutputPath(job, new Path(args[1]));
	job.setJarByClass(Q1b.class);
	job.setOutputKeyClass(GraphPair.class);
	job.setOutputValueClass(NullWritable.class);
	job.setMapperClass(TokenizerMapper.class);
	job.setPartitionerClass(PairPartitioner.class);
	job.setGroupingComparatorClass(PairGroupingComparator.class);
	job.setReducerClass(SecondarySortingReducer.class);
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
