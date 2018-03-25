#include "Lines.h"
#include "input.h"
#include "Random.h"
#include "cuda.h"
#include "host_defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern vector<struct Line> originalConstraints;
extern struct Vertex Solution;
extern int randomSeed;

// Intersection vertex
bool Intersection(struct Line *l1, struct Line *l2, struct Vertex *v1)
{
    if (fabs(l1->a1 * l2->a2 - l2->a1 * l1->a2) < EQUAL_NUM)
    {
        v1 = NULL;
        return false;
    }
    v1->x = -(l1->b * l2->a2 - l2->b * l1->a2) / (l1->a1 * l2->a2 - l2->a1 * l1->a2);
    v1->y = (l1->b * l2->a1 - l2->b * l1->a1) / (l1->a2 * l2->a1 - l1->a1 * l2->a2);
    //printf("Intersection: (%lf, %lf)\n", v1->x, v1->y);
    return true;
}

void Slope(struct Line *l)
{
    if (fabs(l->a2 - 0.0) < EQUAL_NUM)
    {
        if ((l->a1 > 0 && l->a2 < 0) || (l->a1 < 0 && l->a2 > 0))
        {
            l->lslope = FLT_MAX;
        }
        else if ((l->a1 < 0 && l->a2 < 0) || (l->a1 > 0 && l->a2 > 0))
        {
            l->lslope = -FLT_MAX;
        }
        else
        {
            l->lslope = -l->a1 / l->a2;
        }
        return;
    }
    l->lslope = -l->a1 / l->a2;
    return;
}

// Slope line
__device__ void SlopeDevice(struct Line *l)
{
    if (fabs(l->a2 - 0.0) < EQUAL_NUM)
    {
        if ((l->a1 > 0 && l->a2 < 0) || (l->a1 < 0 && l->a2 > 0))
        {
            l->lslope = FLT_MAX;
        }
        else if ((l->a1 < 0 && l->a2 < 0) || (l->a1 > 0 && l->a2 > 0))
        {
            l->lslope = -FLT_MAX;
        }
        else
        {
            l->lslope = -l->a1 / l->a2;
        }
        return;
    }
    l->lslope = -l->a1 / l->a2;
    return;
}

// Compare
int cmp(const void *a, const void *b)
{
    struct Line *aa = (struct Line *)a;
    struct Line *bb = (struct Line *)b;
    return ((aa->lslope > bb->lslope) ? 1 : -1);
}

// Rotation
__global__ void kRotation(struct Line oConstraints[], struct Line lines[], struct Objfunc *object, int *index, int *numG, int *numH)
{
    //__shared__ int numGtemp;
    //__shared__ int numHtemp;



    double thetaArc, thetaDec;

    if (object->c2 == 0 && object->c1 > 0) {
        thetaArc = -PI / 2;
        thetaDec = -90;
    }
    else if (object->c2 == 0 && object->c1 < 0) {
        thetaArc = PI / 2;
        thetaDec = 90;
    }
    else {
        thetaArc = atan(-object->c1 / object->c2);
        thetaDec = atan(-object->c1 / object->c2) * 180 / PI;
    }

    int i;
    double a1Temp, a2Temp, bTemp;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    //printf("offset: %d\n", offset);
    if (offset < (*index)) {

        a1Temp = oConstraints[offset].a1;
        a2Temp = oConstraints[offset].a2;
        bTemp = oConstraints[offset].b;

        lines[offset].a1 = cos(thetaArc) * a1Temp + sin(thetaArc) * a2Temp;
        lines[offset].a2 = cos(thetaArc) * a2Temp - sin(thetaArc) * a1Temp;
        lines[offset].b = bTemp;
        lines[offset].index = offset;

        //printf("%lf\n", lines[offset].a2);

        if (lines[offset].a2 > 0) {
            //__threadfence();
            atomicAdd(numG, 1);
            //(*numG)++;
            //printf("%d", (*numG));
        }
        else if (lines[offset].a2 < 0) {
            //__threadfence();
            atomicAdd(numH, 1);
            //(*numH)++;
            //printf("%d", (*numH));
        }
        else {
            return;
        }

        SlopeDevice(&lines[offset]);
        lines[offset].beingUsed = true;
    }

    //__threadfence();
    __syncthreads();
    __threadfence();

    /*
    if (offset == 2) {
    (*numG) = numGtemp;
    (*numH) = numHtemp;
    }*/

    return;

    /*
    for (i = 0; i < (*index); i += 1) {
    a1Temp = oConstraints[i].a1;
    a2Temp = oConstraints[i].a2;
    bTemp = oConstraints[i].b;

    lines[i].a1 = cos(thetaArc) * a1Temp + sin(thetaArc) * a2Temp;
    lines[i].a2 = cos(thetaArc) * a2Temp - sin(thetaArc) * a1Temp;
    lines[i].b = bTemp;
    lines[i].index = i;

    if (lines[i].a2 > 0) {
    (*numG)++;
    //printf("%d", (*numG));
    }
    else if (lines[i].a2 < 0) {
    (*numH)++;
    //printf("%d", (*numH));
    }
    else {
    (*ret) = false;
    return;
    }

    Slope(&lines[i]);
    lines[i].beingUsed = true;

    }

    if ((*numG) + (*numH) != (*index)) {
    printf("Fatal Error at Rotation()!\n");
    exit(-1);
    }
    */
    return;
}

// Separation - O(n)
bool Separation(struct Line I1[], struct Line I2[], struct Line lines[], int numG, int numH)
{
    int index = numG + numH;
    int i, g = 0, h = 0;
    for (i = 0; i < index; i++) {
        if (lines[i].a2 > 0) {
            I1[g].a1 = -lines[i].a1 / lines[i].a2;
            I1[g].a2 = 1;
            I1[g].b = lines[i].b / lines[i].a2;
            Slope(&I1[g]);
            I1[g].lslope = -I1[g].lslope;
            I1[g].beingUsed = true;
            I1[g].index = lines[i].index;
            //cout << I1[g].index << "\n";
            g++;
        }
        else if (lines[i].a2 < 0) {
            I2[h].a1 = -lines[i].a1 / lines[i].a2;
            I2[h].a2 = 1;
            I2[h].b = lines[i].b / lines[i].a2;
            Slope(&I2[h]);
            I2[h].lslope = -I2[h].lslope;
            I2[h].beingUsed = true;
            I2[h].index = lines[i].index;
            //cout << I2[h].index << "\n";
            h++;
        }
        else {
            printf("%d %lf\n", i, lines[i].a2);
            return false;
        }
    }
    return true;
}

// Make pairs
bool MakePairs(struct Line I1[], struct Line I2[],
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

        if (fabs(I1[g].lslope - I1[gtemp].lslope) < EQUAL_NUM) {
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
        Intersection(&I1[g], &I1[gtemp], p);
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

        //cout << g << " " << gtemp << '\n';
        // printf("Intersection2: (%lf, %lf)\n", p->x, p->y);
        //printf("Value: %lf, %lf\n", I1[g].a1 * p->x + I1[g].b, I1[gtemp].a1 * p->x + I1[gtemp].b);
        (*index)++;
        g++;
    }
    /*
    for (h = 0; h < numH; h += 2) {
    // drop

    pairsH[h / 2].index = *index++;
    pairsH[h / 2].line1 = I2[h];
    pairsH[h / 2].line2 = I2[h + 1];
    Intersection(&I2[h], &I2[h + 1], &pairsH[h / 2].point);
    }*/

    return true;
}

// sg, Sg, sh, Sh
struct Vertex *TestingLine(struct Pair pairsG[], struct Pair pairsH[],
    struct Line I1[], struct Line I2[],
    int numG, int numH, int numDot,
    double *leftBound, double *rightBound)
{
    /*
    for (int i = 0; i < numG; i++) {
    cout << "Line " << i << " : " << I1[i].beingUsed << endl;
    }*/

    // Randomly choose a point
    //cout << "Testing Line" << endl;
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
        if (isFeasible == true && numH != 0) {
            struct Vertex *vTemp = (struct Vertex *)malloc(sizeof(struct Vertex));
            for (int i = 0; i < numH; i++) {
                Intersection(&(I1[onlyOne]), &(I2[i]), vTemp);
                if (vSln->y < vTemp->y) {
                    vSln->x = vTemp->x;
                    vSln->y = vTemp->y;
                }
            }
            printf("sln: %lf %lf\n", vSln->x, vSln->y);
            return vSln;
        }
        else {
            /*
            for (int i = 0; i < numG; i++) {
            cout << "beingUsed: " << I1[i].beingUsed << endl;
            }*/
            cout << "No solution!\n";
            exit(0);
        }
    }

    //int index = round ? 1 : 0;
    double xPrimeG = pairsG[index].point.x;   // x' - xPrime
    double yPrimeG = pairsG[index].point.y;
    double yPrimeH;

    //cout << xPrimeG << '\n';

    // struct Line *sg = (&pairsG[index].line1.a1 < &pairsG[index].line2.a1) ? &pairsG[index].line1 : &pairsG[index].line2;
    // struct Line *Sg = (&pairsG[index].line1.a1 >= &pairsG[index].line2.a1) ? &pairsG[index].line1 : &pairsG[index].line2;
    struct Line *sg = NULL;
    struct Line *Sg = NULL;
    struct Line *sh = NULL;
    struct Line *Sh = NULL;
    // struct Line *sh = (&pairsH[index].line1.a1 < &pairsH[index].line2.a1) ? &pairsH[index].line1 : &pairsH[index].line2;
    // struct Line *Sh = (&pairsH[index].line1.a1 < &pairsH[index].line2.a1) ? &pairsH[index].line1 : &pairsH[index].line2;

    vector<int> linesG;
    vector<int> linesH;

    // Finding g(x') and H(x')
    for (int i = 0; i < numG; i++) {
        if (I1[i].beingUsed == true) {
            if ((fabs(yPrimeG - (I1[i].a1 * xPrimeG + I1[i].b)) > EQUAL_NUM && yPrimeG < (I1[i].a1 * xPrimeG + I1[i].b)) || (sg == NULL || Sg == NULL)) {
                //printf("xPrime yPrime ???: %lf %lf %lf\n", xPrimeG, yPrimeG, (I1[i].a1 * xPrimeG + I1[i].b));



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
            else if (fabs(yPrimeH - (I2[i].a1 * xPrimeG + I2[i].b)) > EQUAL_NUM && yPrimeH > (I2[i].a1 * xPrimeG + I2[i].b)) {
                yPrimeH = I2[i].a1 * xPrimeG + I2[i].b;
                sh = &I2[i];
                Sh = &I2[i];
            }
        }
    }
    if (numH == 0) {
        yPrimeH = yPrimeG + 1000.0;
    }

    // Finding sg - min g(x') && Finding Sg - max g(x')
    /*
    struct Line *sg = &pairsG[0].line1;
    struct Line *Sg = &pairsG[0].line1;
    struct Line *sh = &pairsH[0].line1;
    struct Line *Sh = &pairsH[0].line1;
    */
    for (int i = 0; i < numG; i++) {
        double currentLineValueG = I1[i].a1 * xPrimeG + I1[i].b;
        if (I1[i].beingUsed == false || fabs(currentLineValueG - yPrimeG) >= EQUAL_NUM) {
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
        if (I2[i].beingUsed == false || fabs(currentLineValueH - yPrimeH) >= EQUAL_NUM) {
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
    if (fabs(yPrimeG - yPrimeH) < 1e-6) {
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
            /*
            printf("aaaaaaaaaaa %lf %lf %lf\n", xPrimeG, yPrimeG, yPrimeH);
            cout << sh->a1 << " " << sh->a1 * xPrimeG + sh->b << " " << originalConstraints[sh->index].a1 << '\n';
            cout << Sh->a1 << " " << Sh->a1 * xPrimeG + Sh->b << " " << originalConstraints[Sh->index].a1 << '\n';
            */
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
                    //pairsG[index].line2.beingUsed = false;
                    I1[pairsG[index].index2].beingUsed = false;
                }
                else if (pairsG[index].line1.a1 > pairsG[index].line2.a1) {
                    //pairsG[index].line1.beingUsed = false;
                    I1[pairsG[index].index1].beingUsed = false;
                }
            }
            *rightBound = xPrimeG;
            //cout << "eeeeeeeeeeeeeeeee\n";
            return NULL;
        }
        else if (Sg->a1 < 0) {
            // x* > x'
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
            //cout << "fffffffffffff\n";
            return NULL;
        }
        else if (sg->a1 <= 0 && 0 <= Sg->a1) {
            // x* = x'
            Solution.x = xPrimeG;
            Solution.y = yPrimeG;
            //cout << "hhhhhhhhhhhhhh\n";
            return &(Solution);
        }
    }
    return NULL;
}


///////////////////////////////////////////////////////////////////////////////////
static void HandleError(cudaError_t err,
    const char *file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

bool Rotation(struct Line lines[], struct Objfunc object, int index, int *numG, int *numH)
{
    bool ret;

    // Original Constraints
    struct Line *dev_oConstraints;
    unsigned int size = index * sizeof(struct Line);

    HANDLE_ERROR(cudaMalloc((void**)&dev_oConstraints, size));

    // Lines after rotation
    struct Line *dev_lines;

    HANDLE_ERROR(cudaMalloc((void**)&dev_lines, size));

    // Objective function
    struct Objfunc *dev_object;

    HANDLE_ERROR(cudaMalloc((void**)&dev_object, sizeof(struct Objfunc)));

    // Numbers of lines
    int *dev_index;

    HANDLE_ERROR(cudaMalloc((void**)&dev_index, sizeof(int)));

    // Num of G lines
    int *dev_numG;

    HANDLE_ERROR(cudaMalloc((void**)&dev_numG, sizeof(int)));

    // Num of H lines
    int *dev_numH;

    HANDLE_ERROR(cudaMalloc((void**)&dev_numH, sizeof(int)));

    // Space distribution
    unsigned int DIM = 1 + sqrt(index) / 16;

    dim3 blocks(DIM, DIM);
    dim3 threads(16, 16);

    (*numG) = (*numH) = 0;

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);    //创建Event
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);    //记录当前时间

    cudaEventRecord(stop, 0);    //记录当前时间

    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差


    cudaEventDestroy(start);    //destory the event
    cudaEventDestroy(stop);
    printf("执行时间：%f(ms)\n", time_elapsed);

    // Copy from CPU to GPU
    HANDLE_ERROR(cudaMemcpy(dev_oConstraints, &originalConstraints[0], size, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_object, &object, sizeof(struct Objfunc), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_index, &index, sizeof(int), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_numG, numG, sizeof(int), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_numH, numH, sizeof(int), cudaMemcpyHostToDevice));

    // Kernel function <<<blocks, threads>>>
    kRotation << <blocks, threads >> >(dev_oConstraints, dev_lines, dev_object, dev_index, dev_numG, dev_numH);

    // Copy from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(numG, dev_numG, sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(numH, dev_numH, sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(lines, dev_lines, size, cudaMemcpyDeviceToHost));

    printf("%d %d\n", (*numG), (*numH));

    if ((*numH) + (*numG) != index) {
        ret = false;
    }
    else {
        ret = true;
    }

    return ret;
}




