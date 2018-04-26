package edu.gatech.cse6242;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;

public class Q4 {

  public static class TokenizerMapper extends Mapper<Object,Text,IntWritable,IntWritable>{
    private IntWritable outgoing = new IntWritable(-1);
    private IntWritable ingoing = new IntWritable(1);
    private IntWritable node = new IntWritable();
    
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
      String [] line = value.toString().split("\t");
      if(line.length ==2){
        node.set(Integer.parseInt(line[0]));
        context.write(node,outgoing);
        node.set(Integer.parseInt(line[1]));
        context.write(node,ingoing);
      }
    }
  }

  public static class NodeCountMapper extends Mapper<Object,Text,IntWritable,IntWritable>{
    private IntWritable count = new IntWritable(1);
    private IntWritable diff = new IntWritable();
    
    public void map(Object key, Text value, Context context) throws IOException,InterruptedException{
      String[] line = value.toString().split("\\t");
      if(line.length == 2){
        diff.set(Integer.parseInt(line[1]));
        context.write(diff,count);
      }

    }
  }

  public static class IntSumReuducer extends Reducer<IntWritable,IntWritable, IntWritable, IntWritable>{
    private IntWritable result = new IntWritable();

    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException,InterruptedException{
      int sum = 0;
      for(IntWritable value: values){
        sum += value.get();
      }
      result.set(sum);
      context.write(key,result);
    }
  }



  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job_1 = Job.getInstance(conf, "Q4_1");

    /* TODO: Needs to be implemented */
    job_1.setJarByClass(Q4.class);
    job_1.setMapperClass(TokenizerMapper.class);
    job_1.setCombinerClass(IntSumReuducer.class);
    job_1.setReducerClass(IntSumReuducer.class);
    job_1.setOutputKeyClass(IntWritable.class);
    job_1.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job_1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job_1, new Path("temp"));

    job_1.waitForCompletion(true);
    Job job_2 = Job.getInstance(conf,"Q4_2");
    job_2.setJarByClass(Q4.class);
    job_2.setMapperClass(NodeCountMapper.class);
    job_2.setCombinerClass(IntSumReuducer.class);
    job_2.setReducerClass(IntSumReuducer.class);
    job_2.setOutputKeyClass(IntWritable.class);
    job_2.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job_2,new Path("temp"));
    FileOutputFormat.setOutputPath(job_2,new Path(args[1]));
    System.exit(job_2.waitForCompletion(true) ? 0 : 1);
  }
}
