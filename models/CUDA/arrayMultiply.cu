/*
首先所有线程计算自己对应index的乘法，然后将结果保存到m每个block的共享内存中。然后每个block中指定一个thread将
共享内存的结果归约加和到一起得到该block内thread对应点乘的和。最后再将每个block的共享内存的求和结果最后求和，得到结果。
*/
#include<stdio.h>

const int N = 12 * 256;
const int threadsPerBlock = 256;
const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;

__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock]; //GPU共享内存
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    // 对所有线程块中的线程进行同步，
    __syncthreads();

    //对于归约运算来说，以下代码要求threadPerBlock必须是2的指数
    //基本思想：每个线程将cache[]中的两个值加起来，然后将结果保存回cache[]
    //因为是将两个值归约成一个值，所以每次执行下来，得到的结果数量为开始时的一半。
    //在log2(threadPerBlock)个步骤后，结果就是cache[]的总和。
    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0){//为了防止带来不必要的冗余计算，只让cacheIndex==0线程进行最后的保存操作
        c[blockIdx.x] = cache[0];
        printf("%f\n", cache[0]);
    }
}

int main(void) {
    float a[N], b[N], c, partial_c[N];
    float *dev_a, *dev_b, *dev_partial_c;

    //在GPU上分配内存
    cudaMalloc((void**)&dev_a, N*sizeof(float));
    cudaMalloc((void**)&dev_b, N*sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

    //填充主机内存
    for (int i = 0; i<N; i++){
        a[i] = (float)i;
        b[i] = (float)i;
    }

    //将主机内存复制到GPU上
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    //在CPU上完成最终求和运算
    c = 0;
    for (int i = 0; i<blocksPerGrid; i++){
        c += partial_c[i];
    }
    printf("\n%f\n", c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    delete [] a;
    delete [] b;
    delete [] partial_c;
}
