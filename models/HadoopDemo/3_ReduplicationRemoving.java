/*
数据去重的思想是让原始数据中出现多次的数据在输出文件中只出现一次。非常入门的实例

思路：
1. 在map中直接将每一行数据作为key，NullWritable作为value。因为value没有意义
2. 在reduce端直接将key输出，NullWriable作为输出value（没有意义）
 */

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

public class ReduplicationRemoving{
    static class MyMapper extends Mapper<Object, Text, Text, NullWritable>{
        @Override
        protected void map(Object key, Text value, Context context)
                throws IOException, InterruptedException{
            context.write(value, NullWritable.get());
        }
    }
    static class MyReducer extends Reducer<Text, NullWritable, Text, NullWritable>{
        @Override
        protected void reduce(Text key, Iterable<NullWritable>, Context context){
            context.write(key, NullWritable.get());
        }
    }
    public static void main(String[] args) {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "reduplicationremoving");
        job.setJarByClass(ReduplicationRemoving.class);
        job.setMapperClass(MyMapper.class);
        job.setReduceClass(MyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.addOutputPath(job, new Path(args[1]));
        job.waitForCompletion(true);
    }
}
