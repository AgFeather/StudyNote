import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Mapper;

import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Reducer;

public class MaxTemperatureMapper extends MapReduceBase
    implements Mapper<LongWritable, Text, Text, IntWritable> {//Mapper是一个泛型类型，有四个参数，分别为输入键，输入值，输出键和输出值的类型
//Hadoop 规定了一套可用于网络序列优化的基本类型，不同于Java内置类型，这些可以在org.apache.hadoop.io包找到
//LongWritable相当于Java的Long型，Text相当于String，IntWritable相当于Integer
    private static final int MISSING = 9999;

    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException{
        //map方法需要传入一个键一个值。我们将Text转换成Java的String类型，然后利用substring提取感兴趣的列
        //map方法还提供了一个Context实例来写入输出内容。
        String line = value.toString();
        String year = line.substring(15, 19);
        int airTemperature;
        if (line.charAt(87) == '+') //parseInt doesnot like leading plus signs
            airTemperature = Integer.parseInt(line.substring(88, 92));
        else
            airTemperature = Integer.parseInt(line.substring(87, 92));
        String quality = line.substring(92, 93);
        if (airTemperature != MISSING && quality.matches("[01459]"))
            context.write(new Text(year), new IntWritable(airTemperature));
    }
}


public class MaxTemperatureReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    // map和reduce输入类型必须相同

    @Override
    public void reduce(Text key, Interator<IntWritable> values, Context context)
        throws IOException, InterruptedException {

        int maxValue = Integer.MIN_VALUE;
        while(values.hasNext()){
            maxValue = Math.max(maxValue, values.next().get());
        }
        context.write(key, new IntWritable(maxValue));
    }
}


import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.input.FileOutputFormat;
import org.apache.hadoop.mapreduce.input.FileInputFormat;

public class MaxTemperature {
    public static void main(String[] args) throws Exception {
        if (args.length != 2){
            System.err.println("Usage: MaxTemperature <input path>
                <output path>");
            System.exit(-1);
        }

        Job job = new Job();
        //JobConf对象指定了作业执行规范，授予你对整个作业如何运行的控制权
        job.setJarByClass(MaxTemperature.class);
    //    job.setJarByClass(MaxTemperatureWithCombiner.class);//在map和reduce间加入combiner
        job.setJobName("Max temperature");

        //指定文件的输入和输出路径,可以多次调用addInputPath添加多路径，addOutputPath只能有一个
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.addOutputPath(job, new Path(args[1]));

        //指定要使用的map和reduce类
        job.setMapperClass(MaxTemperatureMapper.class);
    //  job.setCombinerClass(MaxTemperatureReducer.class); //在job中加入combiner函数    
        job.setReduceClass(MaxTemperatureReducer.class);

        //指定map和reduce函数的输出类型
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        //runJob会提交作业并等待它完成，吧进展情况写入控制台。
        System.exit(job.waitForCompletion(true) ? 0:1);
    }
}
