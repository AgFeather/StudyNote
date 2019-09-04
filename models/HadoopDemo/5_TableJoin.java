/*
给定数据表：包含两个字段：1. child，2. parent。要求输出grandchild-grandparent表

设计思路：
1. 首先我们需要进行单表连接，连接的是左表的parent列和右表的child列，且左右表是同一个表。
2. 连接结果中除去连接的两个列就是所需要的祖父-孙子表。

1. MapReduce的shuffle过程会将相同的key值放在一起，所以可以将map结果的key值设成待连接的列。相同连接列的数据自然就在一起了
2. 在map过程中，首先将parent设为key，child设为value进行输出并作为左表；然后将child作为key，parent作为value进行输出作为右表
3. 为了区分输出中的左右表，需要在输出的value中再添加左右表的信息，比如在value的开始处加上字符1表示左表，2表示右表。
4. 然后shuffle会自动连接相同key值的数据。
5. reduce接收到连接结果，其中每个key的value list都包含了grandchild和grandparents的关系。
6. 将左表中的child放入一个数组，右表中的parent放入一个数组，然后对这两个数组求笛卡尔乘积就是最后结果
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

public class TableJoin{
    public static int time = 0;
    //map将输入分割成child和parent，然后正序输出一次作为右表，反序输出一次作为左表
    //需要注意的事在输出的value中必须加上左右表区别标志
    static class MyMapper extends Mapper<Object, Text, Text, Text>{
        @Override
        protected void map(Object key, Text value Context context)
                throws IOException, InterruptedException{
            String childename = new String();
            String parentname = new String();
            String relationtype = new String();
            String line = value.toString();
            int i = 0;
            while(line.cahrAt(i) != ' '){i++;} // 找到表的分隔位置
            String[] values = {line.substring(0, i), line.substring(i+1)};
            if (values[0].compareTo("child") != 0) { // 删除表头
                childename = values[0]; // child对应的值
                parentname = values[1]; // parent对应的值
                relationtype = "1" //左右区分标志，1表示parent作为key，左表
                context.write(new Text(value[1]), new Text(relationtype+""+childname+"$"+parentname));//左表，用$来分别两个字段
                relationtype = "2"; // 2表示child作为key，右表
                context.write(new Text(value[0]), new Text(relationtype+""+childname+"$"+parentname));//右表
            }
        }
    }
    static class MyReducer extends Reducer<Text, Text, Text, Text>{
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            if(time == 0){// 输出表头
                context.write(new Text("grandchild"), new Text("grandparent"));
            }
            ArrayList<String> grandchild = new ArrayList<String>();//用于存放当前连接字段所有左表的child，也就是grandchild
            ArrayList<String> grandparent = new ArrayList<String>();//用于存放当前连接字段所有右表的parent，也就是grandparent
            Iterator ite = values.iterator();
            while(ite.hasNext()){
                String record = ite.next().toString();
                int splitIndex = record.indexof("$")+1;//找到child-parent的分割字符
                char relationtype = record.cahrAt(0);//获取左右表信息，1表示左表，2表示右表
                String childname = record.substring(1, splitIndex);//字段中的child
                String parentname = record.substring(splitIndex);//字段中的parent
                if (relationtype == "1") {//当前表为左表，将child作为grandchild加入list
                    grandchild.add(childname);
                }else{//当前字段为右表，将parent作为grandparent加入到list
                    grandparent.add(parentname);
                }
                if (!grandchild.isEmpty() && !grandparent.isEmpty()) {//对两个list进行遍历，得到笛卡尔乘积输出
                    for (String grandC : grandchild) {
                        for(String grandP : grandparent){
                            context.write(new Text(grandC), new Text(grandP));
                        }
                    }
                }
            }
            time++;
        }
    }
    public static void main(String[] args) {
        Job job = new Job();
        job.setJarByClass(TableJoin.class);
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
