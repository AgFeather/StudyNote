/*


首先介绍一个GroupingComparator组比较器概念：
我们知道，map的输出是一个key-value对，其中value一般是一个number或者string等基本类型，
而reduce接收到的key-value对中的value则是一个list。
完成上述合并操作的就是GroupingComparator，它会对每个key进行比较，对于相同的key，将value放到相同的value list中
注意：Partitioner是mapping端的，GroupingComparator是reduce端的。


有如下数据格式：订单id， 商品id， 成交金额
现在需要找出相同订单id中成交金额最大的一笔交易的信息。



整体实现思路：
1. 首先定义订单bean，需要实现可序列化，和比较方法compareTo，比较规则：订单号不同的，按照订单号比较，相同的，按照金额比较
2. 定义一个Partitioner，根据订单号的hashcode分区，可以保证订单号相同的在同一个分区，以便reduce中接收到同一个订单的全部记录
    同分区中的数据是有序的，这就用到了bean中的比较方法，让订单号相同的记录按照金额大小排序
3. 在map方法输出数据时，key就是bean，value就是null
4. 定义一个GroupingComparator：因为map的结果数据中key是bean，不是普通数据类型，所以需要使用自定义的比较器来分组，就使用bean中的订单号来比较。
    进行比较，前两条数据的订单号相同，放入一组，默认是以第一条记录的key作为这组记录的key。



该代码结构仍然需要重新考虑！！！！
*/

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

public class OrderBean implements WritableComparable<OrderBean>{

    private Text itemid;
    private DoubleWritable amount;

    public OrderBean(){}
    public OrderBean(Text id, DoubleWritable amount){
        this.itemid = id;
        this.amount = amount;
    }
    /*
    私有属性的get和set方法，省略
     */
    public void readFields(DataInput in) throws IOException{
        this.itemid = new Text(in.redaUTF());
        this.amount = new DoubleWritable(in.readDouble());
    }
    public void write(DataOutput out) throws IOException{
        out.writeUTF(itemid.toString());
        out.writeDouble(amount.get());
    }
    public int compareTo(OrderBean o){
        int ret = this.itemid.compareTo(o.getItemid());
        if (ret == 0) {
            ret = -this.amount.compareTo(o.getAmount());
        }
        return ret;
    }
    @Override
    public String toString(){
        return itemid.toString() + "\t" + amount.toString();
    }
}



//定义分区器ItemIdPartitioner
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Partitioner;

public class ItemIdPartitioner extends Partitioner<OrderBean, NullWritable>{
    @Override
    public int getPartition(OrderBean bean, NullWritable value, int numReduceTasks){
        // 相同id的订单bean，会发往相同的partition
        // 产生的分区数会和用户设定的reduce task保持一致
        return (bean.getItemid().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }
}


// 定义比较器
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class MyGroupingComparator extends WritableComparator{
    //对两个Orderbean进行比较，如果订单号相同，则属于同一个key，
    public MyGroupingComparator(){super(OrderBean.class, true);}
    @Override
    public int compare(WritableComparable a, WritableComparable b){
        OrderBean ob1 = (OrderBean)a;
        OrderBean ob2 = (OrderBean)b;
        return ob1.getItemid().comapareTo(ob2.getItemid());
    }
}



// main
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapredcue.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.ouput.FileOutputFormat;

public class GroupSort{
    static class SortMapper extends Mapper<LongWritable, Text, OrderBean, NullWritable>{
        OrderBean bean = new OrderBean();
        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException{
            String line = value.toString();
            Stringp[] fields = line.split(",");
            bean.set(new Text(fields[0]), new DoubleWritable(Double.parseDouble(fields[2])));
            context.write(bean, NullWritable.get());
        }
    }
    static class SortReducer extends Reducer<OrderBean, NullWritable, OrderBean, NullWritable>{
        @Override
        protected void reduce(OrderBean key, Iterable<NullWritable> value, Context context)
                throws IOException, InterruptedException{
            context.write(key, NullWritable.get());
        }
    }


    public static void main(String[] args) {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf);
        job.setJarByClass(GroupSort.class);
        job.setMapperClass(SortMapper.class);
        job.setReducerClass(SortReducer.class);
        job.setOutputKeyClass(OrderBean.class);
        job.setOutputValueClass(OrderBean.class);

        job.setGroupingComparatorClass(MyGroupingComparator.class);
        job.setPartitionerClass(ItemIdPartitioner.class);
        job.setNumReduceTasks(2);

        FileInputFormat.setInputPath(job, new Path(args[0]));
        FileInputFormat.setOutputPath(job, new Path(args[1]));
        job.waitForCompletion(true);
    }
}
