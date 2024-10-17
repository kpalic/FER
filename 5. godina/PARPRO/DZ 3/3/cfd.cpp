#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <sstream>

using namespace std;

#include "arraymalloc.h"
#include "boundary.h"
#include "cfdio.h"
#include "jacobi.h"

// ----------------------------------------------------------------------------------------------------------------------------------------------------
const char *source_jacobi = R"(
__kernel void jacobistep(__global float *psinew, __global const float *psi, int m, int n) {
    int i = get_global_id(0) + 1; // +1 za OKVIR
    int j = get_global_id(1) + 1;
    if (i <= m && j <= n)
        psinew[i*(m+2)+j] = 0.25*(psi[(i-1)*(m+2)+j] + psi[(i+1)*(m+2)+j] + psi[i*(m+2)+j-1] + psi[i*(m+2)+j+1]);
}
)";

const char *source_copy = R"(
__kernel void copy(__global float *psi, __global const float *psitmp, int m, int n) {
     int i = get_global_id(0) + 1;
     int j = get_global_id(1) + 1;
    if (i <= m && j <= n)
        psi[i * (m + 2) + j] = psitmp[i * (m + 2) + j];
}
)";
// ----------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    int printfreq = 1000;  // output frequency
    float error, bnorm;
    float tolerance = 0.0;  // tolerance for convergence. <=0 means do not check

    // main arrays
    float *psi;
    // temporary versions of main arrays
    float *psitmp;

    // command line arguments
    int scalefactor, numiter;

    // simulation sizes
    int bbase = 10;
    int hbase = 15;
    int wbase = 5;
    int mbase = 32;
    int nbase = 32;

    int irrotational = 1, checkerr = 0;

    int m, n, b, h, w;
    int iter;
    int i, j;

    double tstart, tstop, ttot, titer;

    // do we stop because of tolerance?
    if (tolerance > 0) {
        checkerr = 1;
    }

    // check command line parameters and parse them
    if (argc < 3 || argc > 4) {
        printf("Usage: cfd <scale> <numiter>\n");
        return 0;
    }

    scalefactor = atoi(argv[1]);
    numiter = atoi(argv[2]);

    if (!checkerr) {
        printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
    } else {
        printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter, tolerance);
    }

    printf("Irrotational flow\n");

    // Calculate b, h & w and m & n
    b = bbase * scalefactor;
    h = hbase * scalefactor;
    w = wbase * scalefactor;
    m = mbase * scalefactor;
    n = nbase * scalefactor;

    printf("Running CFD on %d x %d grid in serial\n", m, n);

    // allocate arrays
    psi = (float *)malloc((m + 2) * (n + 2) * sizeof(float));
    psitmp = (float *)malloc((m + 2) * (n + 2) * sizeof(float));

    // zero the psi array
    for (i = 0; i < m + 2; i++) {
        for (j = 0; j < n + 2; j++) {
            psi[i * (m + 2) + j] = 0.0;
            psitmp[i * (m + 2) + j] = 0.0;
        }
    }

    // set the psi boundary conditions
    boundarypsi(psi, m, n, b, h, w);

    // compute normalisation factor for error
    bnorm = 0.0;

    for (i = 0; i < m + 2; i++) {
        for (j = 0; j < n + 2; j++) {
            bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
        }
    }
    bnorm = sqrt(bnorm);

    // ----------------------------------------------------------------------------------------------------------------------------------------------------
    // Inicijalizacija OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Red izvođenja
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Učitavanje programa
    cl_program program_jacobi = clCreateProgramWithSource(context, 1, &source_jacobi, NULL, NULL);
    cl_program program_copy = clCreateProgramWithSource(context, 1, &source_copy, NULL, NULL);

    // Prevođenje programa
    cl_int buildStatus_jacobi = clBuildProgram(program_jacobi, 1, &device, NULL, NULL, NULL);
    cl_int buildStatus_copy = clBuildProgram(program_copy, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel_jacobi = clCreateKernel(program_jacobi, "jacobistep", NULL);  // Ne može ime kernel jer je ključna riječ
    cl_kernel kernel_copy = clCreateKernel(program_copy, "copy", NULL);            // Ne može ime kernel jer je ključna riječ

    // Priprema podataka
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (m + 2) * (n + 2) * sizeof(float), psi, NULL);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (m + 2) * (n + 2) * sizeof(float), psitmp, NULL);

    // Argumenti
    clSetKernelArg(kernel_jacobi, 0, sizeof(cl_mem), (void *)&outputBuffer);
    clSetKernelArg(kernel_jacobi, 1, sizeof(cl_mem), (void *)&inputBuffer);
    clSetKernelArg(kernel_jacobi, 2, sizeof(int), &m);
    clSetKernelArg(kernel_jacobi, 3, sizeof(int), &n);

    clSetKernelArg(kernel_copy, 0, sizeof(cl_mem), (void *)&inputBuffer);
    clSetKernelArg(kernel_copy, 1, sizeof(cl_mem), (void *)&outputBuffer);
    clSetKernelArg(kernel_copy, 2, sizeof(int), &m);
    clSetKernelArg(kernel_copy, 3, sizeof(int), &n);

    // Variraj G i L
    size_t globalSize[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
    // size_t localSize[2] = {2048, 2048};                                           // Primer veličine grupe dretvi (local work size)
    // ----------------------------------------------------------------------------------------------------------------------------------------------------

    // begin iterative Jacobi loop
    printf("\nStarting main loop...\n\n");
    tstart = gettime();

    for (iter = 1; iter <= numiter; iter++) {
        // ----------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate psi for next iteration
        clEnqueueNDRangeKernel(queue, kernel_jacobi, 2, NULL, globalSize, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, (m + 2) * (n + 2) * sizeof(float), psitmp, 0, NULL, NULL);
        // ----------------------------------------------------------------------------------------------------------------------------------------------------

        // calculate current error if required
        if (checkerr || iter == numiter) {
            error = deltasq(psitmp, psi, m, n);

            error = sqrt(error);
            error = error / bnorm;
        }

        // quit early if we have reached required tolerance
        if (checkerr) {
            if (error < tolerance) {
                printf("Converged on iteration %d\n", iter);
                break;
            }
        }

        // ----------------------------------------------------------------------------------------------------------------------------------------------------
        // copy back
        clEnqueueNDRangeKernel(queue, kernel_copy, 2, NULL, globalSize, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, (m + 2) * (n + 2) * sizeof(float), psi, 0, NULL, NULL);
        // ----------------------------------------------------------------------------------------------------------------------------------------------------

        // print loop information
        if (iter % printfreq == 0) {
            if (!checkerr) {
                printf("Completed iteration %d\n", iter);
            } else {
                printf("Completed iteration %d, error = %g\n", iter, error);
            }
        }
    }  // iter

    if (iter > numiter) iter = numiter;

    tstop = gettime();

    ttot = tstop - tstart;
    titer = ttot / (double)iter;

    // print out some stats
    printf("\n... finished\n");
    printf("After %d iterations, the error is %g\n", iter, error);
    printf("Time for %d iterations was %g seconds\n", iter, ttot);
    printf("Each iteration took %g seconds\n", titer);

    // output results
    // writedatafiles(psi,m,n, scalefactor);
    // writeplotfile(m,n,scalefactor);

    // ----------------------------------------------------------------------------------------------------------------------------------------------------
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel_jacobi);
    clReleaseProgram(program_jacobi);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(psi);
    free(psitmp);
    printf("... finished\n");
    // ----------------------------------------------------------------------------------------------------------------------------------------------------

    return 0;
}
