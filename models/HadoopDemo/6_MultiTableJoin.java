/*
多表关联：

输入两个表，一个代表工厂表，包含厂名列和地址编号列；另一个表代表地址表，包括地址编号列和地址名列。
要求从输入数据中找出工作名和地址名的对应关系，输出：厂名-地址名表

设计思路：
多表关联和单表关联类似，都是数据库的自然连接，在mapper中用连接列做key，区分好左右表。
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

public class MultiTableJoin{
    public static int time = 0;
//map中先区分输入行属于左表还是右表，然后分别对输入表进行切割
//将连接列作为key值输出，剩余列和左右表标识在value中最后输出。
    static class MyMapper extends Mapper<Object, Text, Text, Text>{
        @Override
        protected void map(Object key, Text value Context context)
                throws IOException, InterruptedException{
            String line = value.toString();
            int i = 0;
            //输文件的首行表头，不做处理
            if(line.contains("factoryname") || line.contains("addID")) return;
            // 找出数据中的分割点
            while(line.charAt(i) >= '9' || line.charAt(i) <= '0') {i++;}
            if (line.charAt(0) >= '9' || line.charAt(0) <= '0') {// 开头不为数字，标识factoryName左表
                int j = i - 1;
                while(line.charAt(j) != ' ') {j--;}
                String values = {line.substring(0, j), line.substring(i)}; // 保存厂名和地址编号
                context.write(new Text(values[1], new Text("1" + values[0])));
            }else{ // 开头为数字，表示地址id，右表
                int j = i + 1;
                while(line.charAt(j) != ' ') {j++;}
                String[] values = {line.substring(0, i+1), line.substring(j)};
                context.write(new Text(values[0]), new Text("2" + values[1]));
            }
        }
    }

    static class MyReducer extends Reducer<Text, Text, Text, Text>{
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            if(time == 0){// 输出表头
                context.write(new Text("factoryname"), new Text("addressname"));
            }
            ArrayList<String> factory = new ArrayList<String>();//用于存放当前连接字段所有左表的值，也就是factoryname
            ArrayList<String> address = new ArrayList<String>();//用于存放当前连接字段所有右表的值，也就是addressname
            Iterator ite = values.iterator();
            while(ite.hasNext()){
                String record = ite.next().toString();
                int splitIndex = record.indexof("$")+1;//找到child-parent的分割字符
                char type = record.cahrAt(0);//获取左右表信息，1表示左表，2表示右表
                if (type == "1") {//当前表为左表，将factory加入到list
                    factory.add(record.substring(1);
                }else{//当前字段为右表，将addressname加入list
                    address.add(record.substring(1));
                }
                if (!factory.isEmpty() && !address.isEmpty()) {//对两个list进行遍历，得到笛卡尔乘积输出
                    for (String f : factory) {
                        for(String a : address){
                            context.write(new Text(f), new Text(a));
                        }
                    }
                }
            }
        }
    }
    public static void main(String[] args) {
        Job job = new Job();
        job.setJarByClass(MultiTableJoin.class);
        job.setOutputKeyClass(.class);
        job.setOutputValueClass()
        job.setMapperClass(MyMapper.class);
        job.setReduceClass(MyReducer.classs);
        FileInputFormat.setInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        boolean res = job.waitForCompletion(true);
        System.exit(res?0:1);
    }
}
