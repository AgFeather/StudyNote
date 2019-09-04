#include<stdio.h>
/*
在GPU上实现矩阵乘法
矩阵a为a[M][N]
矩阵b为[N][S]
矩阵result为[M][S]
*/
__global__ void matmulKernel(const int *a, const int *b, int *result, const int M, const in N, const int S){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x
                + blockIdx.x * blockDim.x + threadIdx.x;// 计算出当前线程的index（对应着result的index）
    if (tid < M * S) {
        int row = tid / S;//本次计算result对应的行值
        int column = tid % S; //本次计算result对应的列值
        result[tid] = 0;
        for(int i = 0;i<N;i++)
            result[tid] += a[row*N + i] * b[i*S + column];//遍历a的第row*N+1行，和b第i*S列对应位相乘
    }
}

/*
使用共享内存__share__实现GPU矩阵乘法可以大大加快运行速度
 */
template<int BLOCK_SIZE>
__global__ void matmulSharedKernel(const int *a, const int *b, int *result, const int M, const in N, const int S){
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    if ((block_y*blockDim.y+thread_y)*gridDim.x*blockDim.x + block_x*blockDim.x+thread_x < M*S){
        const int begin_a = block_y * blockDim.y * N; // 当前thread计算中，矩阵a的起始位置
        const int end_a = begin_a + N-1; // 当前thread计算中，矩阵a的终止位置
        const int step_a = blockDim.x;

        const int begin_b = block_x * blockDim.x; // 当前thread计算中，矩阵b的起始位置
        const int step_b = blockDim.y * S; // 当前thread计算中，矩阵b的终止位置

        int tempResult = 0;

        for(int i = begin_a; int j = begin_b; i<end_a; i += step_a, j += step_b){
            __shared__ int SubMat_A[BLOCK_SIZE][BLOCK_SIZE]
            __shared__ int SubMat_B[BLOCK_SIZE][BLOCK_SIZE]

            SubMat_A[thread_y][thread_x] = a[i + thread_y*N + thread_x];
            SubMat_B[thread_y][thread_x] = b[j + thread_y*S + thread_x];

            __syncthreads();

            for(int k = 0; k<BLOCK_SIZE; k++)
                tempResult += SubMat_A[thread_y][k] * SubMat_B[k][thread_x];

            __syncthreads();
        }
        int begin_result = block_y * blockDim.y * S + begin_b;
        result[begin_result + thread_y*S + thread_x] = tempResult;
    }
}

/*
矩阵乘法的并行运算，每次计算矩阵的一块数据。利用共享内存的共享功能，每次将一块数据保存到共享内存中使得一个线程块同时调用数据进行计算当前块相对应得矩阵乘法结果值。

代码 __shared__ int SubMat_A中的__shared__声明变量SubMat_A为共享内存中保存的变量。然后将数组中的数据提取到变量SubMat_A中保存在共享内存。
__syncthreads()对线程块中的线程进行同步，确保对__shared__进行下面的操作时上面的操作已经完成。
两个for循环完成了当前线程块对应矩阵子块的乘法结果计算。

 */





// 使用纹理内存
/* gpuMatMultWithTextureKernel：GPU下使用texture内存的矩阵乘法
*  result：结果矩阵，表示为result[M][S];
*  M：表示为矩阵A与矩阵result的行数
*  N：表示矩阵A的列数，矩阵B的行数
*  S：表示矩阵B和矩阵result的列数
*/
__global__ void gpuMatMultWithTextureKernel(int * result, const int M, const int N, const int S)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < M * S)
    {
        int a = 0, b = 0;
        int temp_result = 0;
        for (int i = 0; i < N; i++)
        {
            a = tex1Dfetch(texA, y * N + i);
            b = tex1Dfetch(texB, i * S + x);
            temp_result += a * b;
        }
        result[offset] = temp_result;
    }
}

/*
纹理内存的运用与普通内存运用时的算法大致相同，只不过数据是在核函数中调用tex1Dfetch从纹理中提取。

在使用纹理内存时，主要注意的是纹理内存的使用。
首先，需要将输入的数据声明为texture类型的引用。
注意，输入的数据是什么类型，相应的纹理也应该与之一致。并且纹理引用必须声明为文件作用域内的全局变量。

//这些变量将位于GPU上
texture<int> texA;
//二维纹理引用，增加了代表维数的参数2
texture<float, 2> texB;

在为这两个缓冲区分配了GPU内存后，需要通过cudaBindTexture将这些变量绑定到内存缓冲区。这相当于告诉CUDA运行时两件事：

我们希望将指定的缓冲区作为纹理来使用。
我们希望将纹理引用作为纹理的“名字”。
cudaBindTexture(NULL, texA, dev_a, desc, M * N * sizeof(int));
cudaBindTexture(NULL, texB, dev_b, desc, N * S * sizeof(int));

在绑定纹理时，CUDA运行时要求提供一个cudaChannelFormatDesc。此时，需要调用cudaCreateChannelDesc<int>()。

最后，通过cudaUnbindTexture()函数来取消纹理的绑定。


 */
