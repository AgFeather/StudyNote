/*
用CUDA绘制Julia曲线，该例子来着《CUDA by example》。

Julia集的基本算法是：通过一个简单的迭代等式对复平面中的点求值，如果在计算某个点时，迭代等式的计算
结果是发散的，那么这个点就不属于Julia集合。如果迭代等式中计算得到的一系列值都位于某个边界范围内，
那么这个点就属于Julia集合。
迭代等式为：Z(n+1) = Zn^2 + C
 */

#include<stdio.h>
#include<cpu_bitmap.h>
#define DIM 1000

struct cuComplex{
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i; i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r, i+a.i);
	}
}

int main(){
	CPUBitmap bitmap(DIM, DIM);//通过工具库创建一个大小合适的位图图像
	unsigned char *dev_bitmap;//GPU 内存指针
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());//为GPU内存指针开辟内存
	dim3 = grid(DIM,DIM);//声明一个二维线程格，类型dim3表示一个三维数组，可以用于指定启动的线程块的数量
	kernel<<<grid, 1>>>(dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}
__device__ int julia(int x, int y){
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2-x)/(DIM/2);
	float jy = scale * (float)(DIM/2-y)/(DIM/2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	for(int i = 0; i<200; i++){
		a = a*a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}
__global__ void kernel (unsigned char *ptr){
	// 将threadIdx/BlockIdx映射到像素位置
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;//对应的block索引
	//现在计算这个位置上的值
	int juliaValue = julia(x, y);//判断这个点是否属于Julia集，如果位于集合返回1， 否则返回0
	ptr[offset*4+0] = 255 * juliaValue;
	ptr[offset*4+1] = 0;
	ptr[offset*4+2] = 0;
	ptr[offset*4+3] = 255;
}
