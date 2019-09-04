/*
使用CUDA实现数组对应值相加，算是CUDA的hello world。
这段代码在一定程度上进行了优化：可以计算任意长度数组的加和，不限于GPU上block和thread的限制
 */

#include <stdio.h>
#define N 128

__global__ void add(int *a, int *b, int *c){
	int tid = blockDim.x * blockIdx.x  + threadIdx.x;//计算每个thread对应的整体index
	if (tid < N){
		c[tid] = a[tid] + b[tid];
		tid += gridDim.x * blockDim.x;//通过每次位移一个grid所有线程的数量，实现了任意长度数组求和
	}
}

int main(){
	int a[N], b[N], c[N]; //定义主机数组
	int *dev_a, *dev_b, *dev_c; //定义GPU上的内存指针
	//为GPU上的内存分配地址
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_c, N*sizeof(int));
	//初始化主机数组
	for(int i = 0;i<N;i++){
		a[i] = i;
		b[i] = 2*i-3;
	}
	//将主机数组的值复制到GPU内存上
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	//启动核函数，在GPU上开辟(N+127)/128个block，每个block128个thread。其中N+128是为了防止出现N<128开启0个block报错
	add<<<(N+127)/128, 128>>>(dev_a, dev_b, dev_c);
	//将计算的值从GPU内存copy到主机内存
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0;i<N;i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	//释放内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
