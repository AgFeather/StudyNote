import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/*
Hadoop 中的 'Hello world'，词频统计：输入文本数据，统计每个单词的出现次数

MapReduce框架每读取一行数据，就会调用一次map方法，
map处理流程是接收一个key-value对，然后进行逻辑处理，最后向context写入一个key-value对
 */

public class WordCountMapper
        extends Mapper<LongWritable, Text, Text, IntWritable>{//输入key类型，输入value类型，输出key类型，输出value类型
    //Mapper每次接收文件中的一行，然后mapper遍历每个word，向context写入(word, 1)作为输出
    @Override
    protected void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException{
        //得到输入的每一行数据
        String line = value.toString();
        // 通过空格分割
        String[] words = line.split(" ");
        // 循环遍历输出
        for(String word: words){
            context.write(new Text(word), new IntWritable(1));
        }
    }
}
/*
map的输出结果是：(good, 1), (good, 1),(good, 1), (book ,1)
reduce接收到的是：(good, [1,1,1]), (book, [1])
然后reduce将接收到的每个word后的count-list进行就和，得到最终结果：每个词的词频统计
 */
import org.apache.hadoop.mapreduce.Reducer;
public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    @Override
    protected void reduce(Text key, Iteterable<IntWritable> values, Context context)
            throws IOException, InterruptedException{
        Integer count = 0;
        for(IntWritable value:values){
            count += value.get();
        }
        context.write(key, new IntWritable(count));
    }
}


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountMapReduce{
    public static void main(String[] args) throws Exception{
        //创建配置对象
        Configuration conf = new Configuration();
        //创建Job对象
        Job job = Job.getInstance(conf, "wordcount");
        //运行Job的类
        job.setJarByClass(WordCountMapReduce.class);
        //设置mapper和reducer类
        job.setMapperClass(WordCountMapper.class);
        job.setReducerClass(WordCountReducer.class);
        //设置map的输出key value
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        //设置reduce输出的key value
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        //设置输入输出路径
        FileInputFormat.setInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        //提交job到Hadoop
        boolean b = job.waitForCompletion(true);
        if(!b){
            System.out.println("Word Count Task Failed!");
        }
    }
}
