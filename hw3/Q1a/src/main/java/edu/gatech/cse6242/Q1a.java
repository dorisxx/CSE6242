package edu.gatech.cse6242;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.w3c.dom.Text;

public class Q1a {
	
	public static class TokenizerMapper
	extends Mapper<Object,Text, IntWritable, IntWritable>{
		private IntWritable weight = new IntWritable();
		private IntWritable word = new IntWritable();
		
		public void map(Object key, Text value, Context context
				)throws IOException, InterruptedException{
			String[] fields = value.toString().split("\\t");
			if(fields.length==3){
				String source = fields[0];
				String sourceWeight = fields[2];
				weight.set(Integer.parseInt(sourceWeight));
				word.set(Integer.parseInt(source));
				context.write(word,weight);
			}
		}
	}
	
	public static class IntSumReducer
	extends Reducer<IntWritable,IntWritable,IntWritable,IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(IntWritable key, Iterable<IntWritable> values, Context context
				)throws IOException,InterruptedException{
			int max = 0;
			for(IntWritable val:values){
				if(val.get()>max){
					max = val.get();
				}
			}
			result.set(max);
			context.write(key,result);
		}
	}

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Q1a");

    /* TO DO: Needs to be implemented */
    job.setJarByClass(Q1a.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(IntWritable.class);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
