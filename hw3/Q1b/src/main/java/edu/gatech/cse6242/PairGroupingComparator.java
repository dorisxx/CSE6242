package edu.gatech.cse6242;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;


public class PairGroupingComparator extends WritableComparator{
    public PairGroupingComparator(){
        super(GraphPair.class,true);
    }
    @Override
    public int compare(WritableComparable tp1, WritableComparable tp2){
        GraphPair pair1 = (GraphPair)tp1;
        GraphPair pair2 = (GraphPair)tp2;

        return pair1.getReceiver().compareTo(pair2.getReceiver());
    }
}