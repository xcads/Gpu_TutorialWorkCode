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
#include <thrust/extrema.h>
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
		bool operator()(const double x)
	{
		return (x == 0);
	}
};


struct is_prune_or_not
{
	__host__ __device__
		bool operator()(const bool x)
	{
		return x;
	}
};

struct is_feasible
{
	__host__ __device__
		bool operator()(const double a)
	{
		return (a>0);
	}
};

struct is_infeasible
{
	__host__ __device__
		bool operator()(const double a)
	{
		return (a<0);
	}
};


struct is_prune
{
	__host__ __device__
		bool operator()(const double x)
	{
		return (x != DBL_EPSILON);
	}
};

int judgeFeasible(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<int> &t_active,
	int numberofelements,
	double leftBound,
	double rightBound,
	double *sg,
	double *Sg
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
			is_feasible()),
		t_active.end());

	return t_active.size();
}

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

int judgePrune(
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
			is_prune()),
		t_active.end());

	return t_active.size();
}

int findMaxElement(
	thrust::device_vector<double> &t_marker,
	thrust::device_vector<double> &t_active,
	double maxelement
)
{
	t_active.begin() = thrust::max_element(t_marker.begin(), t_marker.end());

	return t_marker.size();
}

int findMinElement(
	thrust::device_vector<double> &t_marker

)
{
	thrust::min_element(t_marker.begin(), t_marker.end());

	return t_marker.size();
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
	double  *lines1g,
	double  *lines2g,
	double  *lines3g,
	int     *arrayJudge,
	double  *vertex_x,
	double  *vertex_y,
	int numberOfLines
)
{
	int x = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	int bid = threadIdx.x + blockIdx.x*blockDim.x;

	if (x < numberOfLines)
	{

		vertex_x[bid] = (lines3[arrayJudge[x]] - lines3[arrayJudge[x+1]]) / (lines1[arrayJudge[x]] - lines1[arrayJudge[x+1]]);
		vertex_y[bid] = (lines1[arrayJudge[x+1]] * lines3[arrayJudge[x]] - lines1[arrayJudge[x ]] * lines3[arrayJudge[x+1]]) / (lines1[arrayJudge[x+1]] - lines1[arrayJudge[x]]);
		lines1g[bid] = lines1[arrayJudge[bid]];
		lines2g[bid] = lines2[arrayJudge[bid]];
		lines3g[bid] = lines3[arrayJudge[bid]];
		if (bid == 328 || bid ==329 || bid ==330) {
			printf("%d,%d:%lf\n",x,bid, lines1[arrayJudge[x]]);
			printf("%d,%d:%lf\n",x,bid, lines2[arrayJudge[x]]);
			printf("%d,%d:%lf\n",x,bid, lines3[arrayJudge[x]]);
			printf("%d,%d:%lf\n",x,bid, lines1[arrayJudge[x+1]]);
			printf("%d,%d:%lf\n",x,bid, lines2[arrayJudge[x+1]]);
			printf("%d,%d:%lf\n",x,bid, lines3[arrayJudge[x+1]]);
			printf("%d,%d:%lf\n",x,bid, vertex_x[bid]);
			printf("%d,%d:%lf\n",x,bid, vertex_y[bid]);

		}
	}

}

__global__ void findLines
(
	double judgeNumber,
	double *lines1,
	double *lines3,
	//	double *lines1new,
	double Value
)
{
	int bid = threadIdx.x + blockDim.x*blockIdx.x;

	if (judgeNumber == lines1[bid])
	{
		Value = lines1[bid] * (judgeNumber)+lines3[bid];
	}

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

	fp = fopen("Coefficient(random).txt", "r");

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
		//lines1.push_back(lineTemp.a2);
		numberOfLines++;

	}
	object.c1 = 1;
	object.c2 = 1;

	//scanf( "%lf%lf", &object.c1, &object.c2);
	//scanf( "%lf%lf", &leftBound, &rightBound);
	leftBound = -10000;
	rightBound = 10000;
	rotationAngleTemp = atan(-object.c1 / object.c2);
	//printf("%lf", rotationAngleTemp);
	struct Line *d_constrainstans;

	//	struct Objfunc *d_object;

	//	int *d_numberOfLines;
	//	double *d_rotationAngle;
	double *d_arrayG;
	double *d_arrayH;
	int size = numberOfLines * sizeof(struct Line);
	//	thrust::device_vector<int> d_lines(numberOfLines);

	HANDLE_ERROR(cudaMalloc((void**)&d_constrainstans, size));


	HANDLE_ERROR(cudaMalloc((void**)&d_arrayG, size));
	HANDLE_ERROR(cudaMalloc((void**)&d_arrayH, size));

	HANDLE_ERROR(cudaMemcpy(d_constrainstans, originalConstraints.data(), size, cudaMemcpyHostToDevice));

	int numberOfBlocks = ceil((float)numberOfLines / block_width);
	thrust::device_vector<double>   lines2;
	thrust::device_vector<double>  d_lines1(numberOfLines);
	thrust::device_vector<double>  d_lines2(numberOfLines);
	thrust::device_vector<double>  d_lines3(numberOfLines);
	thrust::device_vector<double>  d_lines_judge(numberOfLines);
	thrust::device_vector<double>  d_vertex_x(numberOfLines);
	thrust::device_vector<double>  d_vertex_y(numberOfLines);
	thrust::device_vector<double>  d_lines1g(numberOfLines);
	thrust::device_vector<double>  d_lines2g(numberOfLines);
	thrust::device_vector<double>  d_lines3g(numberOfLines);
	thrust::device_vector<double>  d_Sg(numberOfLines);
	thrust::device_vector<double>  d_sg(numberOfLines);
	//<< <numberOfBlocks, block_width >> >
	rotation << <numberOfBlocks, block_width >> >
		(d_constrainstans,
			thrust::raw_pointer_cast(&d_lines1[0]),
			thrust::raw_pointer_cast(&d_lines2[0]),
			thrust::raw_pointer_cast(&d_lines3[0]),
			thrust::raw_pointer_cast(&d_lines_judge[0]),
			rotationAngleTemp,
			numberOfLines);


	double host_array[10000];
	double host_array1[10000];

	int i = seprationG(d_lines_judge, arrayG1, numberOfLines);

	printf("size::%d", i);

	intersection << <numberOfBlocks, block_width >> > (
		thrust::raw_pointer_cast(&d_lines1[0]),
		thrust::raw_pointer_cast(&d_lines2[0]),
		thrust::raw_pointer_cast(&d_lines3[0]),
		thrust::raw_pointer_cast(&d_lines1g[0]),
		thrust::raw_pointer_cast(&d_lines2g[0]),
		thrust::raw_pointer_cast(&d_lines3g[0]),
		thrust::raw_pointer_cast(&arrayG1[0]),
		thrust::raw_pointer_cast(&d_vertex_x[0]),
		thrust::raw_pointer_cast(&d_vertex_y[0]),
		numberOfLines
		);




	cudaMemcpy(host_array, thrust::raw_pointer_cast(&d_vertex_x[0]), 10000*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array1, thrust::raw_pointer_cast(&d_vertex_y[0]), 10000 * sizeof(double), cudaMemcpyDeviceToHost);

	printf("328:%lf\n", host_array[328]);
	printf("328:%lf\n", host_array1[328]);
	printf("329:%lf\n", host_array[329]);
	printf("329:%lf\n", host_array1[329]);
	printf("330:%lf\n", host_array[330]);
	printf("330:%lf\n", host_array1[330]);
	//printf("%lf\n", host_array[0]);
	//printf("%lf\n", host_array[1]);
	//printf("%lf\n", host_array[2]);
	//printf("%lf\n", host_array[3]);
	//printf("%lf\n", host_array[4]);
	/*FILE *fpWrite = fopen("Coefficient_new.txt", "w");
	if (fpWrite == NULL)
	{
		return 0;
	}
	for (int i = 0; i<10000; i++)
		fprintf(fpWrite, "%lf\t%lf\n ", host_array[i],host_array1[i]);
	fclose(fpWrite);
*/
	//thrust::device_vector<double>  sg(sizeof(double));
	//thrust::pair<double *, double *> result = thrust::minmax_element((thrust::raw_pointer_cast(d_lines1g.begin()), thrust::raw_pointer_cast(d_lines1g.end())));


	//md_lines1g[randomIndex];



	//printf("%lf", sg);
	//printf("%lf", Sg);
	//	double Sg;
	//double sh;
	//double Sh;
	////
	////	double judgeSg = Sg - sg;
	////double judgeSh = sh - Sh;

	//double sgValue;
	//double SgValue;
	//double shValue;
	//double ShValue;

	//double xPrime_g = 0;
	//double yPrime_g = 0;




	////feasilbe
	////do the paper method to judege the feasible or infesilble 
	////and ma

	//double *optimalAnswer_x = NULL;
	//double *optimalAnswer_y = NULL;
	////loop (should do 4 things)
	////1. feasible or infeasible
	////2. find the orentation about the optimal answer
	////3. refresh the left and right bound
	////
	////4. drop some lines
	//while (1) {
	//	// is feasibel
	//	//x'<x*

	//	//d_Sg.begin()= thrust::max_element(d_lines1g.begin(), d_lines1g.end());
	//	//	d_sg.begin()= thrust::min_element(d_lines1g.begin(), d_lines1g.end());
	//	//findMaxElement(d_lines1g);
	//	//	double sg = thrust::min_element(thrust::raw_pointer_cast(d_lines1g.begin()), thrust::raw_pointer_cast(d_lines1g.end()));
	//	//	double Sg = thrust::max_element(thrust::raw_pointer_cast(d_lines1g.begin()), thrust::raw_pointer_cast(d_lines1g.end()));
	//	//findMinElement(d_lines1g);
	//	thrust::device_vector<double>::iterator Sg1 = thrust::max_element(d_lines1g.begin(), d_lines1g.end());
	//	unsigned int position = Sg1 - d_lines1g.begin();
	//	double Sg = d_lines1g[position];

	//	thrust::device_vector<double>::iterator sg1 = thrust::min_element(d_lines1g.begin(), d_lines1g.end());
	//	unsigned int position1 = sg1 - d_lines1g.begin();
	//	double sg = d_lines1g[position1];

	//	printf("%lf", sg);
	//	printf("%lf", Sg);
	//	int sizeOfG = arrayG1.size();
	//	srand((unsigned)time(NULL));
	//	int randomIndex = rand() % (sizeOfG);

	//	////  
	//	/*	findLines << <numberOfBlocks, block_width >> >(
	//	d_lines1g[randomIndex],
	//	thrust::raw_pointer_cast(&d_lines1g[0]),
	//	thrust::raw_pointer_cast(&d_lines3g[0]),
	//	sgValue);*/
	//	findLines << <numberOfBlocks, block_width >> > (
	//		d_lines1g[randomIndex],
	//		thrust::raw_pointer_cast(&d_lines1g[0]),
	//		thrust::raw_pointer_cast(&d_lines3g[0]),
	//		SgValue);
	//	//x' h value

	//	//findLines << <numberOfBlocks, block_width >> > (
	//	//	d_lines1g[randomIndex],
	//	//	thrust::raw_pointer_cast(&d_lines1g[0]),
	//	//	thrust::raw_pointer_cast(&d_lines3g[0]),
	//	//    shValue);
	//	/*findLines << <numberOfBlocks, block_width >> > (
	//	d_lines1g[randomIndex],
	//	thrust::raw_pointer_cast(&d_lines1g[0]),
	//	thrust::raw_pointer_cast(&d_lines3g[0]),
	//	ShValue);*/

	//	if ((SgValue - ShValue) < DBL_EPSILON)
	//	{
	//		if (sg > 0 && sg >= Sh) {
	//			if (sh != Sh)
	//				sh = DBL_EPSILON;
	//			if (sg != Sg)
	//				Sg = DBL_EPSILON;
	//		}
	//		rightBound = d_vertex_x[randomIndex];
	//		judgePrune(d_lines1g, arrayG1, arrayG1.size());
	//	}
	//	else  if (Sg < 0 && Sg < sh) {
	//		if (sh != Sh)
	//			Sh = DBL_EPSILON;
	//		if (sg != Sg)
	//			sg = DBL_EPSILON;
	//		leftBound = d_vertex_x[randomIndex];
	//		judgePrune(d_lines1g, arrayG1, arrayG1.size());
	//	}
	//	else {
	//		*optimalAnswer_x = d_vertex_x[randomIndex];
	//		*optimalAnswer_y = d_vertex_y[randomIndex];
	//	}
	//	if (SgValue - ShValue > 0) {
	//		if (sg > Sh) {
	//			if (sh != Sh) {
	//				sh = DBL_EPSILON;
	//			}
	//			if (sg != Sg) {
	//				Sg = DBL_EPSILON;
	//			}
	//		}
	//		else {
	//			if (randomIndex % 2 == 0) {
	//				if (d_lines1g[randomIndex] < d_lines1g[randomIndex - 1]) {
	//					d_lines1g[randomIndex - 1] = DBL_EPSILON;
	//				}
	//				else if (d_lines1g[randomIndex] > d_lines1g[randomIndex - 1]) {
	//					d_lines1g[randomIndex] = DBL_EPSILON;
	//				}

	//			}
	//			else {
	//				if (d_lines1g[randomIndex + 1] < d_lines1g[randomIndex]) {
	//					d_lines1g[randomIndex] = DBL_EPSILON;
	//				}
	//				else if (d_lines1g[randomIndex + 1] > d_lines1g[randomIndex]) {
	//					d_lines1g[randomIndex + 1] = DBL_EPSILON;
	//				}
	//			}
	//			rightBound = d_vertex_x[randomIndex];
	//		}
	//		if (Sg < sh) {
	//			if (sh != Sh) {
	//				Sh = DBL_EPSILON;
	//			}
	//			if (sg != Sg) {
	//				sg = DBL_EPSILON;
	//			}
	//			else {
	//				if (randomIndex % 2 == 0) {
	//					if (d_lines1g[randomIndex] < d_lines1g[randomIndex - 1]) {
	//						d_lines1g[randomIndex] = DBL_EPSILON;
	//					}
	//					else if (d_lines1g[randomIndex] > d_lines1g[randomIndex - 1]) {
	//						d_lines1g[randomIndex - 1] = DBL_EPSILON;
	//					}

	//				}
	//				else {
	//					if (d_lines1g[randomIndex + 1] < d_lines1g[randomIndex]) {
	//						d_lines1g[randomIndex + 1] = DBL_EPSILON;
	//					}
	//					else if (d_lines1g[randomIndex + 1] > d_lines1g[randomIndex]) {
	//						d_lines1g[randomIndex] = DBL_EPSILON;
	//					}
	//				}
	//				leftBound = d_vertex_x[randomIndex];
	//			}
	//			if ((sg - Sh) <= 0 && (Sg - sh) >= 0) {
	//				printf("No Answer can be found");
	//				exit(0);
	//			}
	//		}

	//		if (SgValue < ShValue) {
	//			if (sg > 0) {
	//				if (sg != Sg) {
	//					Sg = DBL_EPSILON;
	//				}
	//				else {
	//					if (randomIndex % 2 == 0) {
	//						if (d_lines1g[randomIndex] < d_lines1g[randomIndex - 1]) {
	//							d_lines1g[randomIndex - 1] = DBL_EPSILON;
	//						}
	//						else if (d_lines1g[randomIndex] > d_lines1g[randomIndex - 1]) {
	//							d_lines1g[randomIndex] = DBL_EPSILON;
	//						}

	//					}
	//					else {
	//						if (d_lines1g[randomIndex + 1] < d_lines1g[randomIndex]) {
	//							d_lines1g[randomIndex] = DBL_EPSILON;
	//						}
	//						else if (d_lines1g[randomIndex + 1] > d_lines1g[randomIndex]) {
	//							d_lines1g[randomIndex + 1] = DBL_EPSILON;
	//						}
	//					}
	//					rightBound = d_vertex_x[randomIndex];
	//				}
	//				if (Sg < 0) {
	//					if (sg != Sg) {
	//						sg = DBL_EPSILON;
	//					}
	//					else {
	//						if (randomIndex % 2 == 0) {
	//							if (d_lines1g[randomIndex] < d_lines1g[randomIndex - 1]) {
	//								d_lines1g[randomIndex] = DBL_EPSILON;
	//							}
	//							else if (d_lines1g[randomIndex] > d_lines1g[randomIndex - 1]) {
	//								d_lines1g[randomIndex - 1] = DBL_EPSILON;
	//							}

	//						}
	//						else {
	//							if (d_lines1g[randomIndex + 1] < d_lines1g[randomIndex]) {
	//								d_lines1g[randomIndex + 1] = DBL_EPSILON;
	//							}
	//							else if (d_lines1g[randomIndex + 1] > d_lines1g[randomIndex]) {
	//								d_lines1g[randomIndex] = DBL_EPSILON;
	//							}
	//						}
	//						leftBound = d_vertex_x[randomIndex];
	//					}
	//					if (sg <= 0 && Sg >= 0) {

	//						*optimalAnswer_x = d_vertex_x[randomIndex];
	//						*optimalAnswer_y = d_vertex_y[randomIndex];
	//					}
	//				}
	//			}
	//		}
	//		judgePrune(d_lines1g, arrayG1, arrayG1.size());
	//		if (optimalAnswer_x != NULL&& optimalAnswer_y != NULL)
	//			break;
	//	}

	//	printf("Stop1");
	//	printf("%lf %lf", optimalAnswer_x, optimalAnswer_y);
	//	/*thrust::sort(d_vertex_x.begin(), d_vertex_x.end());
	//	cudaMemcpy(host_array, thrust::raw_pointer_cast(&d_vertex_x[0]), 50 * sizeof(double), cudaMemcpyDeviceToHost);*/

	//	//	thrust::sort(d_vertex_x[0] , d_vertex_x[arrayG1.size()-1]);
	//	//	printf("%lf", d_vertex_x[arrayG1.size() / 2]);

	//	//	printf("%lf", d_vertex_x[arrayG1.size() / 2]);
	//	//	cudaMemcpy(host_array, thrust::raw_pointer_cast(&d_vertex_x[0]), size, cudaMemcpyDeviceToHost);

	//	//printf("%lf", host_array[0]);
	//	//printf("%lf", host_array[1]);
	//	//printf("%lf", host_array[2]);
	//	//printf("%lf", host_array[3]);
	//	//printf("%lf", host_array[5000]);
	//	/*gpuMedfind << <numberOfBlocks, block_width >> >
	//	(
	//	thrust::raw_pointer_cast(&d_vertex_x[0]),
	//	medianNumber,
	//	numberOfLines
	//	);*/

	//	//cudaMemcpy(host_array, thrust::raw_pointer_cast(&medianNumber[0]),  sizeof(double), cudaMemcpyDeviceToHost);
	//	//cudaMemcpy(medianNumber, d_medianNumber, sizeof(double), cudaMemcpyDeviceToHost);

	//	//thrust::copy(arrayG1.begin(), arrayG1.end(), output.begin());
	//	////
	//	//printf("%lf  ", output[0]);
	//	//cudaFree(d_constrainstans);
	//	//cudaFree(d_lines);
	//	//cudaFree(d_numberOfLines);

	//	//cudaFree(d_rotationAngle);
	//}
}

