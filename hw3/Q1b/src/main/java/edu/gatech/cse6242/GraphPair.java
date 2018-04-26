package edu.gatech.cse6242;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;


public class GraphPair implements Writable, WritableComparable<GraphPair>{
    private Text receiver = new Text(); 
    private IntWritable weight = new IntWritable(); 
    private IntWritable sender = new IntWritable(); 
    
    public GraphPair(){
    }

    public static GraphPair read(DataInput in) throws IOException {
        GraphPair gPair = new GraphPair();
        gPair.readFields(in);
        return gPair;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        receiver.write(out);
        weight.write(out);
        sender.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        receiver.readFields(in);
        weight.readFields(in);
        sender.readFields(in);
    }


    //the setter methods
    public void setReceiver(String s){
        receiver.set(s);
    }
    public void setWeight(int w){
        weight.set(w);
    }
    public void setSender(int s){
        sender.set(s);
    }
    //the getter methods
    public Text getReceiver(){
        return receiver;
    }
    public IntWritable getWeight(){
        return weight;
    }
    public IntWritable getSender(){
        return sender;
    }
    
    @Override
    public int compareTo(GraphPair pair){
        Integer this_r = Integer.parseInt(this.receiver.toString());
        Integer that_r = Integer.parseInt(pair.getReceiver().toString());

        int compareValue = this_r.compareTo(that_r);
        if(compareValue == 0){
            compareValue = (-1)*(this.weight.compareTo(pair.getWeight()));
        }
        if(compareValue ==0){
            compareValue = this.sender.compareTo(pair.getSender());
        }
        return compareValue;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GraphPair that = (GraphPair) o;

        if (receiver != null ? !receiver.equals(that.receiver) : that.receiver != null) return false;
        if (weight != null ? !weight.equals(that.weight) : that.weight != null) return false;
        if (sender != null ? !sender.equals(that.sender) : that.sender != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = receiver != null ? receiver.hashCode() : 0;
        result = 31 * result + (sender != null ? sender.hashCode() : 0);
        return result;
    }
}