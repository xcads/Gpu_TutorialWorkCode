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
//
//struct is_zero
//{
//	__host__ __device__
//		bool operator()(const int x)
//	{
//		return (x == 0);
//	}
//};

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

	//printf("%lf", t_active[0]);
	//printf("%lf", t_active[1]);
	//printf("%lf", t_active[2]);
	//printf("%lf", t_active[3]);
	//printf("%lf", t_active[4]);
	//printf("%lf", t_active[5]);
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
	double rotationAngle, 
	int numberOfLines)
{
	int i=0;
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
		//if (x == 0) {
		//	printf("%lf\n    ", a1Temp);
		//	printf("%lf\n    ", a2Temp);
		//	printf("%lf\n    ", bTemp);
		//	printf("lines2:");
		//	printf("%lf\n    ", lines1[x]);
		//	printf("%lf\n    ", lines2[x]);
		//	printf("%lf\n    ", lines3[x]);
		//}
		
	}

}

//int separation
//(
//	thrust::device_vector<double>  &d_lines1,
//	thrust::device_vector<double>  &d_lines2,
//	thrust::device_vector<double>  &d_lines3
//)
//{
//	int numberOfLines = 0;
//	double leftBound, rightBound;
//	double aTemp, bTemp, cTemp;
//	struct Objfunc object;
//	double rotationAngleTemp;
//	
//	FILE* fp;
//
//	thrust::host_vector<double>   lines1(numberOfLines);
//	thrust::device_vector<int> arrayG1(numberOfLines);
//
//	fp = fopen("Coefficient.txt", "r");
//
//	while (1) {
//		fscanf_s(fp, "%lf%lf%lf", &aTemp, &bTemp, &cTemp);
//		if (aTemp == 0.0 && bTemp == 0.0 && cTemp == 0.0) {
//			break;
//		}
//		struct Line lineTemp;
//		lineTemp.a1 = aTemp;
//		lineTemp.a2 = bTemp;
//		lineTemp.b = cTemp;
//		originalConstraints.push_back(lineTemp);
//		lines1.push_back(lineTemp.a2);
//
//		numberOfLines++;
//
//	}
//
//	scanf("%lf%lf", &object.c1, &object.c2);
//
//
//	rotationAngleTemp = atan(-object.c1 / object.c2);
//
//	struct Line *d_constrainstans;
//
//	struct Objfunc *d_object;
//
//	int *d_numberOfLines;
//	double *d_rotationAngle;
//	double *d_arrayG;
//	double *d_arrayH;
//	int size = numberOfLines * sizeof(struct Line);
//
//
//	HANDLE_ERROR(cudaMalloc((void**)&d_constrainstans, size));
//	HANDLE_ERROR(cudaMalloc((void**)&d_constrainstans, size));
//
//	HANDLE_ERROR(cudaMalloc((void**)&d_arrayG, size));
//	HANDLE_ERROR(cudaMalloc((void**)&d_arrayH, size));
//
//	HANDLE_ERROR(cudaMemcpy(d_constrainstans, originalConstraints.data(), size, cudaMemcpyHostToDevice));
//
//	int numberOfBlocks = ceil((float)numberOfLines / block_width);
//	rotation << <numberOfBlocks, block_width >> >
//		(d_constrainstans,
//			thrust::raw_pointer_cast(&d_lines1[0]),
//			thrust::raw_pointer_cast(&d_lines2[0]),
//			thrust::raw_pointer_cast(&d_lines3[0]),
//			rotationAngleTemp,
//			numberOfLines);
//
//	printf("%lf", d_lines2[0]);
//	return 0;
//}


int main()
{
	/*int numberOfLines = 0;
	double leftBound, rightBound;
	double aTemp, bTemp, cTemp;
	struct Objfunc object;
	double rotationAngleTemp;

	FILE* fp;
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
		numberOfLines++;

	}

	thrust::device_vector<double>  d_lines1(numberOfLines);
	thrust::device_vector<double>  d_lines2(numberOfLines);
	thrust::device_vector<double>  d_lines3(numberOfLines);
	separation(d_lines1,d_lines2,d_lines3);*/
	int numberOfLines = 0;
	double leftBound, rightBound;
	double aTemp, bTemp, cTemp;
	struct Objfunc object;
	double rotationAngleTemp;

	//double *arrayG=(double*)malloc(numberOfLines * sizeof(double));
	//double *arrayH=(double*)malloc(numberOfLines * sizeof(double));
	FILE* fp;
	//thrust::device_vector<double> d_lines1(numberOfLines);
	thrust::host_vector<double>   lines1(numberOfLines);
	thrust::device_vector<int> arrayG1(numberOfLines);

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
	//	printf("%lf\n ",lines1[2]);
		numberOfLines++;
	
	}
	//for (int i = 0; i < numberOfLines; i++)
	//{
	//	printf("%lf\n ", lines1[i]);
	//}
	//printf("%lf%lf%lf\n", originalConstraints[0].a1, originalConstraints[0].a2, originalConstraints[0].b);
	scanf( "%lf%lf", &object.c1, &object.c2);
	//scanf( "%lf%lf", &leftBound, &rightBound);

	rotationAngleTemp = atan(-object.c1 / object.c2);
	//printf("%lf", rotationAngleTemp);
	struct Line *d_constrainstans;
	//double *d_lines1;
	//double *d_lines2;
	//double *d_lines3;
	struct Objfunc *d_object;

	int *d_numberOfLines;
	double *d_rotationAngle;
	double *d_arrayG;
	double *d_arrayH;
	int size = numberOfLines * sizeof(struct Line);
//	thrust::device_vector<int> d_lines(numberOfLines);

	HANDLE_ERROR(cudaMalloc((void**)&d_constrainstans, size));
	//HANDLE_ERROR(cudaMalloc((void**)&d_lines1, size));
	//HANDLE_ERROR(cudaMalloc((void**)&d_lines2, size));
	//HANDLE_ERROR(cudaMalloc((void**)&d_lines3, size));

	HANDLE_ERROR(cudaMalloc((void**)&d_arrayG, size));
	HANDLE_ERROR(cudaMalloc((void**)&d_arrayH, size));

	HANDLE_ERROR(cudaMemcpy(d_constrainstans, originalConstraints.data(), size, cudaMemcpyHostToDevice));

	int numberOfBlocks=ceil((float)numberOfLines/block_width);
	thrust::device_vector<double>   lines2;
	thrust::device_vector<double>  d_lines1(numberOfLines);
	thrust::device_vector<double>  d_lines2(numberOfLines);
	thrust::device_vector<double>  d_lines3(numberOfLines);
	/*thrust::device_vector<double>   d_lines2(numberOfLines);
	thrust::device_vector<double>   d_lines3(numberOfLines);*/
	//
	//
	//<< <numberOfBlocks, block_width >> >
	rotation << <numberOfBlocks, block_width >> >
				(d_constrainstans,
					thrust::raw_pointer_cast(&d_lines1[0]),
					thrust::raw_pointer_cast(&d_lines2[0]),
					thrust::raw_pointer_cast(&d_lines3[0]),
					rotationAngleTemp,
					numberOfLines);

	int host_array[4];
	//cudaMemcpy(host_array, thrust::raw_pointer_cast(&d_lines2[0]), 4*sizeof(double), cudaMemcpyDeviceToHost);

	//printf("%lf   ", host_array[0]);
	//printf("%lf   ", host_array[1]);
	//printf("%lf   ", host_array[2]);
	//printf("%lf   ", host_array[3]);


	int i = seprationG(d_lines2, arrayG1, numberOfLines);
	//printf("%d", i);
	 
	 cudaMemcpy(host_array, thrust::raw_pointer_cast(&arrayG1[0]), 4 * sizeof(int), cudaMemcpyDeviceToHost);
		 printf("%d   ", host_array[0]);
		 printf("%d   ", host_array[1]);
		 printf("%d   ", host_array[2]);
		 printf("%d   ", host_array[3]);

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
