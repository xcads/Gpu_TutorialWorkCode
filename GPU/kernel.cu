#include "cuda.h"
#include "host_defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct_rotation.h"
#include "book.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <time.h>
#include <string.h>
#include <limits.h>	
#include <float.h>


#define numberOfLines1 10000
#define block_width 64
vector<struct Line> originalConstraints;

struct is_positive
{
	__host__ __device__
		bool operator()(const double x)
	{
		return (x > 0);
	}
};

struct is_negative
{
	__host__ __device__
		bool operator()(const double x)
	{
		return (x < 0);
	}
};

struct is_zero
{
	__host__ __device__
		bool operator()(const int x)
	{
		return (x == 0);
	}
};

int seprationG(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<int> &t_active,
	int numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			is_positive()),
		t_active.end());

	return t_active.size();
}

int seprationH(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<int> &t_active,
	int numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			is_negative()),
		t_active.end());

	return t_active.size();
}

int seprationZero(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<int> &t_active,
	int numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			is_zero()),
		t_active.end());

	return t_active.size();
}
int seprationLinesG(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<int> &t_active,
	int numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			is_zero()),
		t_active.end());

	return t_active.size();
}


double rotationAngle(struct Objfunc *object)
{
	double rotationAngle;

	if (object->c2 == 0 && object->c1 > 0) {
		rotationAngle = -PI / 2;
	}
	else if (object->c2 == 0 && object->c1 < 0) {
		rotationAngle = PI / 2;
	}
	else {
		rotationAngle = atan(-object->c1 / object->c2);
	}

	return rotationAngle;
}
//,double *arrayG,double  arrayH[]
__global__ void rotation(
	struct Line constraints[],
	double  *lines1,
	double  *lines2,
	double  *lines3,
	double  *lines_judge,
	double rotationAngle, 
	int numberOfLines)
{

	double a1Temp, a2Temp, bTemp;
	//thrust::device_vector<double>  lines1;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	thrust::device_vector<double> lines1;
	if (x < (numberOfLines)) {
		a1Temp = constraints[x].a1;
		a2Temp = constraints[x].a2;
		bTemp = constraints[x].b;
		
		lines1[x] = (cos(rotationAngle) * a1Temp) + (sin(rotationAngle) * a2Temp);
		lines2[x] = (cos(rotationAngle) * a2Temp) - (sin(rotationAngle) * a1Temp);
		lines3[x] = bTemp;
		lines_judge[x] = lines2[x];
		
		/*if(x<10)
		printf("%lf\n", lines2[x]);*/
		lines1[x] = -lines1[x] / lines2[x];
		lines3[x] = lines3[x] / lines2[x];
		lines2[x] = 1;
		
		
	}

}

__global__ void intersection
(
	double  *lines1,
	double  *lines2,
	double  *lines3,
	int     *arrayJudge,
	double  *vertex_x,
	double  *vertex_y,
	int numberOfLines
)
{
	int x = 2*(threadIdx.x + blockIdx.x * blockDim.x);

	//int j= x *2;
	
	if (x < numberOfLines)
		if (x < 11) {
			//printf("jjj:%d\n", j);
			//if (j == 0)
			//{
			////	printf("J:%d: &lf", j,lines3[j]);
			//	vertex_x[x] = (lines3[j] - lines3[j + 1]) / (lines1[j] - lines1[j+1]);
			//	vertex_y[x] = (lines1[j + 1] * lines3[j] - lines1[j] * lines2[j + 1]) / (lines1[j + 1] - lines1[j]);
			//	//vertex_x[x] = (lines3[j] * lines2[j + 1] - lines3[j + 1] * lines2[j]) / (lines1[j] * lines2[j + 1] - lines1[j + 1] * lines2[j]);
			//	//vertex_y[x] = (lines3[j] * lines1[j + 1] - lines3[j + 1] * lines1[j]) / (lines2[j] * lines1[j + 1] - lines1[j] * lines2[j + 1]);
			//	printf("j===0");
			//	printf("%lf\n", lines3[j]);
			//	printf("%lf\n", lines3[j + 1]);
			//	printf("%lf\n", lines1[j]);
			//	printf("%lf\n", lines1[j + 1]);
			//	printf("  intersection:  %lf  %lf\n ", vertex_x[x], vertex_y[x]);
			//}
			//else
			//{
			//	printf("J:%d:", j);
				vertex_x[arrayJudge[x]] = (lines3[arrayJudge[x-1]] - lines3[arrayJudge[x]]) / (lines1[arrayJudge[x - 1]] - lines1[arrayJudge[x]]);
				vertex_y[x] = (lines1[arrayJudge[x]] * lines3[arrayJudge[x - 1]] - lines1[arrayJudge[x - 1]] * lines2[arrayJudge[x]]) / (lines1[arrayJudge[x]] - lines1[arrayJudge[x - 1]]);
				/*vertex_x[x] = (lines3[j-1] * lines2[j ] - lines3[j] * lines2[j-1]) / (lines1[j-1] * lines2[j ] - lines1[j] * lines2[j-1]);
				vertex_y[x] = (lines3[j-1] * lines1[j ] - lines3[j ] * lines1[j-1]) / (lines2[j-1] * lines1[j ] - lines1[j-1] * lines2[j ]);
*/
			//	printf("%lf\n", lines3[j - 1]);
				//printf("   intersection:  %lf  %lf\n ", vertex_x[x], vertex_y[x]);
			//}

		}
		


}

inline void swap(double a, double b) {
	int temp = a;
	a = b;
	b = temp;
}

__device__ inline void swapGpu(double a, double b) {
	int temp = a;
	a = b;
	b = temp;
}

__global__ void gpuMedfind(double *entries, double med, int numEntries) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 3) + threadIdx.x;

	if (i + 2 * blockDim.x < numEntries) {
		int list[3];
		list[0] = entries[i], list[1] = entries[i + blockDim.x], list[2] = entries[i + 2 * blockDim.x];
		if (list[1] < list[0])
			swapGpu(list[1], list[0]);
		if (list[2] < list[0])
			swapGpu(list[2], list[0]);
		if (list[2] < list[1])
			swapGpu(list[2], list[1]);

		sdata[tid] = list[1];
	}

	__syncthreads();

	for (int s = blockDim.x / 3; s > 0; s /= 3) {
		if (tid < s && tid + 2 * s < blockDim.x) {
			int list[3];
			list[0] = sdata[tid], list[1] = sdata[tid + s], list[2] = sdata[tid + 2 * s];
			if (list[1] < list[0])
				swapGpu(list[1], list[0]);
			if (list[2] < list[0])
				swapGpu(list[2], list[0]);
			if (list[2] < list[1])
				swapGpu(list[2], list[1]);

			sdata[tid] = list[1];
		}

		__syncthreads();
	}

	med = sdata[0];
}

__global__ void JudgeLines
(

)
{

}

int main()
{
	
	int numberOfLines = 0;
	double leftBound, rightBound;
	double aTemp, bTemp, cTemp;
	struct Objfunc object;
	double rotationAngleTemp;

	FILE* fp;

	thrust::host_vector<double>   lines1(numberOfLines);
	thrust::device_vector<int>  arrayG1(numberOfLines);

	fp = fopen("Coefficient.txt", "r");
	
	while (1) {
		fscanf_s(fp, "%lf%lf%lf", &aTemp, &bTemp, &cTemp);
		if (aTemp == 0.0 && bTemp == 0.0 && cTemp == 0.0) {
			break;
		}
		struct Line lineTemp;
		lineTemp.a1 = aTemp;
		lineTemp.a2 = bTemp;
		lineTemp.b = cTemp;
		originalConstraints.push_back(lineTemp);
		lines1.push_back(lineTemp.a2);
		numberOfLines++;
	
	}
	object.c1 = 1;
	object.c2 = 1;

	//scanf( "%lf%lf", &object.c1, &object.c2);
	//scanf( "%lf%lf", &leftBound, &rightBound);

	rotationAngleTemp = atan(-object.c1 / object.c2);
	//printf("%lf", rotationAngleTemp);
	struct Line *d_constrainstans;

	struct Objfunc *d_object;

	int *d_numberOfLines;
	double *d_rotationAngle;
	double *d_arrayG;
	double *d_arrayH;
	int size = numberOfLines * sizeof(struct Line);
//	thrust::device_vector<int> d_lines(numberOfLines);

	HANDLE_ERROR(cudaMalloc((void**)&d_constrainstans, size));


	HANDLE_ERROR(cudaMalloc((void**)&d_arrayG, size));
	HANDLE_ERROR(cudaMalloc((void**)&d_arrayH, size));

	HANDLE_ERROR(cudaMemcpy(d_constrainstans, originalConstraints.data(), size, cudaMemcpyHostToDevice));

	int numberOfBlocks=ceil((float)numberOfLines/block_width);
	thrust::device_vector<double>   lines2;
	thrust::device_vector<double>  d_lines1(numberOfLines);
	thrust::device_vector<double>  d_lines2(numberOfLines);
	thrust::device_vector<double>  d_lines3(numberOfLines);
	thrust::device_vector<double>  d_lines_judge(numberOfLines);
	thrust::device_vector<double>  d_vertex_x(numberOfLines/2);
	thrust::device_vector<double>  d_vertex_y(numberOfLines / 2);
	//<< <numberOfBlocks, block_width >> >
	rotation << <numberOfBlocks, block_width >> >
				(d_constrainstans,
					thrust::raw_pointer_cast(&d_lines1[0]),
					thrust::raw_pointer_cast(&d_lines2[0]),
					thrust::raw_pointer_cast(&d_lines3[0]),
					thrust::raw_pointer_cast(&d_lines_judge[0]),
					rotationAngleTemp,
					numberOfLines);

	double host_array[1];
	
	int i = seprationG(d_lines_judge, arrayG1, numberOfLines);

	/*thrust::device_vector<double>  d_lines1_g(numberOfLines);1
	thrust::device_vector<double>  d_lines2_g(numberOfLines);
	thrust::device_vector<double>  d_lines3_g(numberOfLines);*/

//	thrust::device_vector<double>  medianNumber(1);
	double medianNumber=0;
	/*double *d_medianNumber;
	cudaMalloc((void**)&d_medianNumber, sizeof(double));
	cudaMemcpy(d_medianNumber, medianNumber, sizeof(double), cudaMemcpyHostToDevice);*/

	intersection << <numberOfBlocks, block_width >> > (
		thrust::raw_pointer_cast(&d_lines1[0]),
		thrust::raw_pointer_cast(&d_lines2[0]),
		thrust::raw_pointer_cast(&d_lines3[0]),
		thrust::raw_pointer_cast(&arrayG1[0]),
		thrust::raw_pointer_cast(&d_vertex_x[0]),
		thrust::raw_pointer_cast(&d_vertex_y[0]),
		numberOfLines
		);
	//
	gpuMedfind << <numberOfBlocks, block_width >> >
		(
		thrust::raw_pointer_cast(&d_vertex_x[0]),
		medianNumber,
		numberOfLines
		);


//	cudaMemcpy(host_array, thrust::raw_pointer_cast(&medianNumber[0]),  sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(medianNumber, d_medianNumber, sizeof(double), cudaMemcpyDeviceToHost);
	printf("%lf\n   ", medianNumber);
	//printf("%lf\n   ", host_array[1]);
	//printf("%lf\n   ", host_array[2]);
	//printf("%lf\n   ", host_array[3]);
/*
	 cudaMemcpy(host_array, thrust::raw_pointer_cast(&arrayG1[0]), 4 * sizeof(int), cudaMemcpyDeviceToHost);
		 printf("%d   ", host_array[0]);
		 printf("%d   ", host_array[1]);
		 printf("%d   ", host_array[2]);
		 printf("%d   ", host_array[3]);*/

	// 
	//printf("%lf   ", d_lines2[0]);
	//printf("%lf   ", d_lines2[1]);
	//printf("%lf   ", d_lines2[2]);
	//printf("%lf   ", d_lines2[3]);
	//thrust::host_vector<double> output(numberOfLines);
	//HANDLE_ERROR(cudaMemcpy(d_lines2.data(),arrayH,  size, cudaMemcpyDeviceToHost));
	 //printf("t_marker:%lf   ", arrayH[0]);

	//thrust::copy(arrayG1.begin(), arrayG1.end(), output.begin());
	// 
//	printf("%lf  ", output[0]);
	
	
	// 
	//HANDLE_ERROR()
	//HANDLE_ERROR(cudaMemcpy(arrayG, d_arrayG, size, cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(arrayH, d_arrayH, size, cudaMemcpyDeviceToHost));

	//printf("%d\n", arrayG[0]);

	//printf("%d\n", arrayH[0]);

	//

	//	printf("%lf %lf %lf", originalConstraints[1].a1, originalConstraints[1].a2, originalConstraints[1].b);
	//
	//seprationG(lines2, arrayG1, numberOfLines);

	
	//thrust::copy(arrayG1.begin(), arrayG1.end(), output.begin());
	////
	//printf("%lf  ", output[0]);
	//cudaFree(d_constrainstans);
	//cudaFree(d_lines);
	//cudaFree(d_numberOfLines);

	//cudaFree(d_rotationAngle);
}

  