/*
给定数据格式为一个int数字/每行。
要求输出中每行有两个间隔的数字，第二个数字代表原始数据，第一个数字代表这个原始数据的位次

流程：
1. 在mapper中将输入的string类型的数据转换为int
2. MapReduce会自动根据kye值对mapper的输出进行排序，我们只需要使用这个默认排序即可
3. 在reducer中定义一个static int变量表示顺序，每次reduce写出一个数据，就将这个顺序值++

 */

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.Writable;
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

public class DataSort{
    static class SortMapper extends Mapper<Object, Text, IntWritable, IntWritable>{
        private static IntWritable data = new IntWritable();
        // map将输入中string类型的value转换为int，作为输出的key
        @Override
        protected void map(Object key, Text value Context context)
                throws IOException, InterruptedException{
            String line = value.toString();
            data.set(Integer.parserInt(line));
            context.write(data, new IntWritable(1));
        }
    }
    static class SortRedcuer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable>{
        private static IntWritable linenum = new IntWritable(1); //表示排序
        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException{
            for (IntWritable val: values ) {
                context.write(linenum, key);
                linenum = new IntWritable(linenum.get() + 1);
            }
        }
    }
    public static void main(String[] args) {
        Job job = new Job();
        job.setJarByClass(DataSort.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setMapperClass(SortMapper.class);
        job.setReduceClass(SortRedcuer.classs);
        FileInputFormat.setInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        boolean res = job.waitForCompletion(true);
        System.exit(res?0:1);
    }
}
