/*
给定数据是每个电话号每个月的 1.上传流量使用量；2.下载流量使用量；
Assignment1: 统计每个电话号所有月份的上传/下载/总使用流量总和
Assignment2: 将统计结果根据省份分别输出到不同文件夹中




1. 在这个数据集中，所有的流量信息是用一个字符串类型Text表示的，所以我们可以定义一个bean来表示各个流量信息。
    所以第一步就是将原始数据中以string类型保存的流量数据转换成一个bean
2. reduce接收一个手机号表示的key，以及这个手机号对应的bean对象集合
 */

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.Writable;

public class FlowBean implements Writable{
    private long upFlow;
    private long dFlow;
    private long sumFlow;

    public FlowBean(){}
    public FlowBean(long upFlow, long dFlow){
        this.upFlow = upFlow;
        this.dFlow = dFlow;
        this.sumFlow = upFlow + dFlow;
    }
    /*
    定义各个私有变量的set方法和get方法，略
     */
    public void write(DataOutput out) throws IOException{
        out.writeLong(upFlow);
        out.writeLong(dFlow);
        out.writeLong(sumFlow);
    }
    public void readFields(DataInput in) throws IOException{
        upFlow = in.readLong();
        dFlow = in.readLong();
        sumFlow = in.readLong();
    }
    @Override
    public String toString(){
        return upFlow + "\t" + dFlow + "\t" + sumFlow;
    }
}


import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FlowCount{
    static class FlowCountMapper extends Mapper<LongWritable, Text, Text, FlowBean>{
        @Override
        protected void map(LongWritable key, Text value Context context)
                throws IOException, InterruptedException{
            String line = value.toString();
            String fields = line.split("\t");
            String phoneNbr = fields[0];
            long upFlow = Long.parseLong(fields[1]);
            long dFlow = Long.parseLong(fields[2]);
            context.write(new Text(phoneNbr), new FlowBean(upFlow, dFlow));
        }
    }
    static class FlowCountRedcuer extends Reducer<Text, FlowBean, Text, FlowBean>{
        @Override
        protected void reduce(Text key, Iterable<FlowBean> value, Context context)
                throws IOException, InterruptedException{
            long count_upFlow = 0;
            long count_dFlow = 0;
            for(FlowBean bean: value){
                count_dFlow += bean.getDFlow();
                count_upFlow += bean.getUpFlow();
            }
            context.write(key, new FlowBean(count_upFlow, count_dFlow));
        }
    }
    public static void main(String[] args) {
        Job job = new Job();
        job.setJarByClass(FlowCount.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FlowBean.class);
        job.setMapperClass(FlowCountMapper.class);
        job.setReduceClass(FlowCountRedcuer.classs);
        FileInputFormat.setInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        boolean res = job.waitForCompletion(true);
        System.exit(res?0:1);
    }
}




/*
Assignment2
在上个例子的统计需要基础上添加一个新需求：按省份统计，不同省份的手机号放到不同的文件里。
例如137表示属于河北，138属于河南，那么在结果输出时，他们分别在不同的文件中。

思路：
map和reduce的处理思路与上例相同，这里需要多做2步：
自定义一个分区器Partitioner
根据手机号判断属于哪个分区。有几个分区就有几个reducetask，每个reducetask输出一个文件，这样不同分区中的数据就写入了不同的结果文件中。
在main函数中指定使用我们自定义的Partitioner即可
 */

import java.util.HashMap;
import org.apache.hadoop.io.Text;
import org.apache.mapreduce.Partitioner;

public class ProvincePartitioner extends Partitioner<Text, FlowBean>{
    /*
    这段代码是本示例的重点，其中定义了一个hashmap，假设其是一个数据库，定义了手机号和分区的关系。
    getPartition取得手机号的前缀，到数据库中获取区号，如果没在数据库中，就指定其为“其它分区”（用4代表）
     */
    public static HashMap<String, Integer> provinceDict = new HashMap<String, Integer>();
    static{
        provinceDict.put("137", 0);
        provinceDict.put("133", 1);
        provinceDict.put("138", 2);
        provinceDict.put("135", 3);
    }
    @Override
    public int getPartition(Text key, FlowBean value, int numPartitions){
        String prefix = key.toString().substring(0,3);
        Integer provinceId = provinceDict.get(prefix);
        return provinceId==null?4:provinceId;
    }
}

//同时，在main函数中指定我们自定义的数据分区器
job.setPartitionerClass(ProvincePartitioner.class);
job.setNumReduceTasks(5);
