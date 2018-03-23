
#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

using namespace std;

#define PI   3.14159265358979323846264338327950288419716939937510       
#define RANDOM_SEED     7
#define RANDOM_PARA     2000

int randomSeed = RANDOM_SEED;
// LINE structure : Constraints
struct Line {
	// a1x + a2y >= b
	double a1, a2, b;
	double lslope;
	bool beingUsed;

	int index;
};

// Object Function
struct Objfunc {
	// xd = c1x + c2y
	double c1, c2;
};

// VERTEX structure
struct Vertex {
	double x, y;
};

// PAIR structure
struct Pair {
	int index;
	int index1, index2;
	Line line1, line2;
	Vertex point;
	bool beingUsed;
};

typedef struct Line Line;
typedef struct Objfunc Objfunc;
typedef struct Vertex Vertex;

vector<struct Line> originalConstraints;

struct Vertex Solution;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


int getRandomSeed(int *randomSeed)
{
	(*randomSeed) += RANDOM_PARA;
	if ((*randomSeed) == 0) {
		(*randomSeed) = (int)time(NULL) % RANDOM_SEED + RANDOM_SEED * RANDOM_SEED;
	}
	return ((*randomSeed) * (int)time(NULL));
}

double getRandomDouble(int *randomSeed, double leftBound, double rightBound)
{
	srand(getRandomSeed(randomSeed));
	return (leftBound + (rightBound - leftBound) * rand() / (RAND_MAX));
}

int getRandomInt(int *randomSeed, int Bound)
{
	srand(getRandomSeed(randomSeed));
	return (rand() % Bound);
}

__global__ void transformation(struct Line lines[], struct Objfunc object, int index, int *g, int *h,int symbol)
{
	double transformationAngle;	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (object.c2 == 0 && object.c1 > 0) {
		transformationAngle = -PI / 2;
	}
	else if (object.c2 == 0 && object.c1 < 0) {
		transformationAngle = PI / 2;
	}
	else {
		transformationAngle = atan(-object.c1 / object.c2);
	}

	double a1Temp, a2Temp, bTemp;
	(*g) = 0;
	(*h) = 0;
	a1Temp = originalConstraints[tid].a1;
	a2Temp = originalConstraints[tid].a2;
	bTemp = originalConstraints[tid].b;

	lines[tid].a1 = cos(transformationAngle) * a1Temp + sin(transformationAngle) * a2Temp;
	lines[tid].a2 = cos(transformationAngle) * a2Temp - sin(transformationAngle) * a1Temp;
	lines[tid].b = bTemp;
	lines[tid].index = tid;
	
//	__syncthreads();

	if (lines[tid].a2 > 0) {
		(*g)++;
	}
	else if (lines[tid].a2 < 0) {
		(*h)++;
	}
	else {
		symbol=0;
	}

	//Slope(&lines[i]);
	lines[tid].beingUsed = true;

	if ((*g) + (*h) != index) {
		printf("Fatal Error at Rotation()!\n");
		exit(-1);
	}
	symbol = 1;
}

__device__ bool Slope(struct Line *l)
{
	if (fabs(l->a2 - 0.0) < DBL_EPSILON)
	{
		if ((l->a1 > 0 && l->a2 < 0) || (l->a1 < 0 && l->a2 > 0))
		{
			l->lslope = DBL_MAX;
		}
		else if ((l->a1 < 0 && l->a2 < 0) || (l->a1 > 0 && l->a2 > 0))
		{
			l->lslope = -DBL_MAX;
		}
		else
		{
			l->lslope = -l->a1 / l->a2;
		}
		return false;
	}
	l->lslope = -l->a1 / l->a2;
	return true;
}

__global__ void Intersection(struct Line *l1, struct Line *l2, struct Vertex *v1,int symbol)
{
	if (abs(l1->a1 * l2->a2 - l2->a1 * l1->a2) < DBL_EPSILON)
	{
		v1 = NULL;
		symbol=0;
	}
	v1->x = -(l1->b * l2->a2 - l2->b * l1->a2) / (l1->a1 * l2->a2 - l2->a1 * l1->a2);
	v1->y = (l1->b * l2->a1 - l2->b * l1->a1) / (l1->a2 * l2->a1 - l1->a1 * l2->a2);
	symbol = 1;
}


__global__ void Segmentation(struct Line I1[], struct Line I2[], struct Line lines[], int numG, int numH,int symbol)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int index = numG + numH;
	int i, g = 0, h = 0;

	if (lines[tid].a2 > 0) {
		I1[tid].a1 = -lines[tid].a1 / lines[tid].a2;
		I1[tid].a2 = 1;
		I1[tid].b = lines[tid].b / lines[tid].a2;
		Slope(&I1[tid]);
		I1[tid].lslope = -I1[tid].lslope;
		I1[tid].beingUsed = true;
		I1[tid].index = lines[tid].index;
		g++;
	}
		if (lines[tid].a2 < 0) {
			I2[tid].a1 = -lines[tid].a1 / lines[tid].a2;
			I2[tid].a2 = 1;
			I2[tid].b = lines[tid].b / lines[tid].a2;
			Slope(&I2[tid]);
			I2[tid].lslope = -I2[tid].lslope;
			I2[tid].beingUsed = true;
			I2[tid].index = lines[tid].index;
			//cout << I2[h].index << "\n";
		}
		//	__syncthreads();
		else {
			symbol=0;
		}
		symbol = 1;
	}
	
__device__ bool MakePairs(struct Line I1[], struct Line I2[],
	struct Pair pairsG[], struct Pair pairsH[],
	int numG, int numH, int *index,
	double leftBound, double rightBound)
{
	int g, gtemp;
	(*index) = 0;
	for (g = 0; g < numG; g += 1) {
		// drop
		if (I1[g].beingUsed == false) {
			continue;
		}
		for (gtemp = g + 1; gtemp < numG; gtemp++) {
			if (I1[gtemp].beingUsed == true) {
				break;
			}
		}
		if (gtemp == numG) break;

		if (abs(I1[g].lslope - I1[gtemp].lslope) < DBL_EPSILON) {
			if (I1[g].b > I1[gtemp].b) {
				I1[gtemp].beingUsed = false;
				g = g - 1;
			}
			else {
				I1[g].beingUsed = false;
				g = gtemp - 1;
			}

			continue;
		}
		struct Vertex *p = (struct Vertex *)malloc(sizeof(struct Vertex));
		int symbol;
		Intersection(&I1[g], &I1[gtemp], p,symbol);
		if (p->x < leftBound || p->x > rightBound) {
			if (p->x < leftBound && (I1[g].lslope > I1[gtemp].lslope)) {
				I1[gtemp].beingUsed = false;
				g = g - 1;
			}
			else if (p->x < leftBound && (I1[g].lslope < I1[gtemp].lslope)) {
				I1[g].beingUsed = false;
				g = gtemp - 1;
			}
			else if (p->x > rightBound && (I1[g].lslope < I1[gtemp].lslope)) {
				I1[gtemp].beingUsed = false;
				g = g - 1;
			}
			else if (p->x > rightBound && (I1[g].lslope > I1[gtemp].lslope)) {
				I1[g].beingUsed = false;
				g = gtemp - 1;
			}
			continue;
		}
		pairsG[(*index)].index = (*index);
		pairsG[(*index)].line1 = I1[g];
		pairsG[(*index)].index1 = g;
		pairsG[(*index)].line2 = I1[gtemp];
		pairsG[(*index)].index2 = gtemp;
		pairsG[(*index)].point.x = p->x; pairsG[(*index)].point.y = p->y;

		(*index)++;
		g++;
	}
	return true;
}


struct Vertex *TestingLine(struct Pair pairsG[], struct Pair pairsH[],
	struct Line I1[], struct Line I2[],
	int numG, int numH, int numDot,
	double *leftBound, double *rightBound)
{

	int index = (numDot == 0) ? 0 : (getRandomInt(&randomSeed, numDot));

	//printf("%d %d\n", index, numDot);

	if (numDot == 0) {
		int onlyOne = 0;
		bool isFeasible = false;
		struct Vertex *vSln = (struct Vertex *)malloc(sizeof(struct Vertex));
		vSln->y = -FLT_MAX;
		for (onlyOne = 0; onlyOne < numG; onlyOne++) {
			if (I1[onlyOne].beingUsed == true) {
				isFeasible = true;
				break;
			}
		}
		int symbol;
		if (isFeasible == true && numH != 0) {
			struct Vertex *vTemp = (struct Vertex *)malloc(sizeof(struct Vertex));
			for (int i = 0; i < numH; i++) {
				Intersection(&(I1[onlyOne]), &(I2[i]), vTemp,symbol);
				if (vSln->y < vTemp->y) {
					vSln->x = vTemp->x;
					vSln->y = vTemp->y;
				}
			}
			printf("sln: %lf %lf\n", vSln->x, vSln->y);
			return vSln;
		}
		else {

			cout << "No solution!\n";
			exit(0);
		}
	}

	double xPrimeG = pairsG[index].point.x;   // x' - xPrime
	double yPrimeG = pairsG[index].point.y;
	double yPrimeH;

	struct Line *sg = NULL;
	struct Line *Sg = NULL;
	struct Line *sh = NULL;
	struct Line *Sh = NULL;

	vector<int> linesG;
	vector<int> linesH;

	// Finding g(x') and H(x')
	for (int i = 0; i < numG; i++) {
		if (I1[i].beingUsed == true) {
			if ((abs(yPrimeG - (I1[i].a1 * xPrimeG + I1[i].b)) >  DBL_EPSILON && yPrimeG < (I1[i].a1 * xPrimeG + I1[i].b)) || (sg == NULL || Sg == NULL)) {

				yPrimeG = I1[i].a1 * xPrimeG + I1[i].b;
				sg = &I1[i];
				Sg = &I1[i];
			}
		}
	}
	for (int i = 0; i < numH; i++) {
		if (I2[i].beingUsed == true) {
			if (sh == NULL || Sh == NULL) {
				sh = &I2[i];
				Sh = &I2[i];
				yPrimeH = I2[i].a1 * xPrimeG + I2[i].b;
			}
			else if (abs(yPrimeH - (I2[i].a1 * xPrimeG + I2[i].b)) >  DBL_EPSILON && yPrimeH > (I2[i].a1 * xPrimeG + I2[i].b)) {
				yPrimeH = I2[i].a1 * xPrimeG + I2[i].b;
				sh = &I2[i];
				Sh = &I2[i];
			}
		}
	}
	if (numH == 0) {
		yPrimeH = yPrimeG + 1000.0;
	}

	for (int i = 0; i < numG; i++) {
		double currentLineValueG = I1[i].a1 * xPrimeG + I1[i].b;
		if (I1[i].beingUsed == false || abs(currentLineValueG - yPrimeG) >= DBL_EPSILON) {
			continue;
		}

		if (I1[i].a1 < sg->a1) {
			sg = &I1[i];
		}
		if (I1[i].a1 > Sg->a1) {
			Sg = &I1[i];
		}
	}
	// Finding sh - min h(x') && Finding Sh - max h(x')
	for (int i = 0; i < numH; i++) {
		double currentLineValueH = I2[i].a1 * xPrimeG + I2[i].b;
		if (I2[i].beingUsed == false || abs(currentLineValueH - yPrimeH) >= DBL_EPSILON) {
			continue;
		}

		if (I2[i].a1 < sh->a1) {
			sh = &I2[i];
		}
		if (I2[i].a1 > Sh->a1) {
			Sh = &I2[i];
		}
	}

	// Is feasible
	if (abs(yPrimeG - yPrimeH) < 1e-6) {
		if (sg->a1 > 0 && sg->a1 >= Sh->a1) {
			// x* < x'
			if (sh != Sh) {
				sh->beingUsed = false;
			}
			if (sg != Sg) {
				Sg->beingUsed = false;
			}
			*rightBound = xPrimeG;
			//cout << "cccccccccc\n";
			return NULL;
		}
		else if (Sg->a1 < 0 && Sg->a1 <= sh->a1) {
			// x* > x'
			if (sh != Sh) {
				Sh->beingUsed = false;
			}
			if (sg != Sg) {
				sg->beingUsed = false;
			}
			*leftBound = xPrimeG;
			//cout << "dddddddddddddd\n";
			return NULL;
		}
		else {
			// x* = x'
			Solution.x = xPrimeG;
			Solution.y = yPrimeG;
			//cout << "gggggggggggggggggg\n";
			return &(Solution);
		}
	}
	else if (yPrimeG > yPrimeH) {   // infeasible
		if (sg->a1 > Sh->a1) {
			// x* < x'
			if (sh != Sh) {
				sh->beingUsed = false;
			}
			if (sg != Sg) {
				Sg->beingUsed = false;
			}

			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					//pairsG[index].line2.beingUsed = false;
					I1[pairsG[index].index2].beingUsed = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
					//pairsG[index].line1.beingUsed = false;
					I1[pairsG[index].index1].beingUsed = false;
				}
			}
			*rightBound = xPrimeG;
		
			return NULL;
		}
		else if (Sg->a1 < sh->a1) {
			// x* > x'
			if (sh != Sh) {
				Sh->beingUsed = false;
			}
			if (sg != Sg) {
				sg->beingUsed = false;
			}

			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
					//pairsG[index].line1.beingUsed = false;
					I1[pairsG[index].index1].beingUsed = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
					//pairsG[index].line2.beingUsed = false;
					I1[pairsG[index].index2].beingUsed = false;
				}
			}
			*leftBound = xPrimeG;
			//printf("bbbbbbbbbbbbbbb\n");
			return NULL;
		}
		else if ((sg->a1 - Sh->a1) <= 0 && 0 <= (Sg->a1 - sh->a1)) {
			// no feasible
			printf("No feasible solution!\n");
			exit(0);
			return NULL;
		}
	}
	else if (yPrimeG < yPrimeH) {   // feasible
		if (sg->a1 > 0) {
			// x* < x'
			if (sg != Sg) {
				Sg->beingUsed = false;
			}
			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
				
					I1[pairsG[index].index2].beingUsed = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
			
					I1[pairsG[index].index1].beingUsed = false;
				}
			}
			*rightBound = xPrimeG;
		
			return NULL;
		}
		else if (Sg->a1 < 0) {
	
			if (sg != Sg) {
				sg->beingUsed = false;
			}
			else {
				if (pairsG[index].line1.a1 < pairsG[index].line2.a1) {
			
					I1[pairsG[index].index1].beingUsed = false;
				}
				else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
		
					I1[pairsG[index].index2].beingUsed = false;
				}
			}
			*leftBound = xPrimeG;
			return NULL;
		}
		else if (sg->a1 <= 0 && 0 <= Sg->a1) {
			// x* = x'
			Solution.x = xPrimeG;
			Solution.y = yPrimeG;
			return &(Solution);
		}
	}
	return NULL;
}
void LinearProgramming(void)
{
	int indexRecord = 0;
	int numGRecord;
	int numHRecord;
	int indexPair;
	double leftBound, rightBound;
	double aTemp, bTemp, cTemp;
	bool judge = false;
	struct Objfunc object;

	//int round = 0;
	FILE* fp;

	fp = fopen("coffcient.txt", "r+");

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
		indexRecord++;
	}
	fscanf_s(fp, "%lf%lf", &object.c1, &object.c2);
	fscanf_s(fp, "%lf%lf", &leftBound, &rightBound);

	cout << "lalala\n";

	struct Line *lines = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Line *I1 = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Line *I2 = (struct Line *)malloc(indexRecord * sizeof(struct Line));
	struct Pair *pairG = (struct Pair *)malloc(indexRecord * sizeof(struct Pair));
	struct Pair *pairH = (struct Pair *)malloc(indexRecord * sizeof(struct Pair));
	struct Vertex *sln = NULL;
	int symbol1;
	transformation(lines, object, indexRecord, &numGRecord, &numHRecord,symbol1);
	if (symbol1 == false) {
		printf("Fatal Error at LinearProgramming() - Rotation()!\n");
		exit(-1);
	}
	int symbol;
	Segmentation(I1, I2, lines, numGRecord, numHRecord,symbol);
	if (symbol == 0) {
		printf("Fatal Error at LinearProgramming() - Segmentation()!\n");
		exit(-1);
	}

	//cout << numGRecord << " " << numHRecord << '\n';
	/*
	for (int i = 0; i < numGRecord; I++) {
	printf("")
	}
	*/

	while (1) {
		judge = MakePairs(I1, I2, pairG, pairH, numGRecord, numHRecord, &indexPair, leftBound, rightBound);
		if (judge == false) {
			printf("Fatal Error at LinearProgramming() - MakePairs()!\n");
			exit(-1);
		}

		sln = TestingLine(pairG, pairH, I1, I2, numGRecord, numHRecord, indexPair, &leftBound, &rightBound);
		//cout << leftBound << " " << rightBound << '\n';
		if (sln != NULL) {
			break;
		}
	}

	printf("sln: %lf %lf\n", sln->x, sln->y);



	fclose(fp);
	return;

}

__global__ void findOptimal()
{
	
}
int main()
{
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}
	
	LinearProgramming();

	return 0;
}