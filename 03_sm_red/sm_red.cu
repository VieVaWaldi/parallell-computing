#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DTYPE float

/**
   Single Vector - Shared Memory
*/

__global__ void kernelSMReduce(DTYPE *vect, DTYPE *result, int size)
{
   int tid = threadIdx.x;
   int id = tid + blockIdx.x * blockDim.x;

   __shared__ float sm[512];
   sm[tid] = vect[id];
   __syncthreads();

   for (int i=blockDim.x/2; i>0; i/=2)
   {
      if (tid < i)
      {
         sm[tid] += sm[tid+i];
      }
      __syncthreads();
   }

   if (tid == 0)
   {
      *result = sm[0];
   }
}

void hostSMReduce(DTYPE *vect, DTYPE *result, int size)
{
   DTYPE sum = 0.0;
   for (int i=0; i<size; i++)
      sum += vect[i];
   *result = sum; 
}

void fillVect(DTYPE *vect, int size)
{
   for (int i=0; i<size; i++)
      vect[i] = 1.0;
}

bool checkResult(DTYPE *vect, DTYPE *result, int size)
{
   DTYPE sum = 0.0;
   for (int i=0; i<size; i++)
      sum += vect[i];
   printf("sum = %f, result = %f\n", sum, *result);
   return sum == *result;
}

/*
   Main Routine: 
   Input: i,[threads]
   Berechnet Vector Reduce Sum auf GPU and Host, using shared memory
   n=1024*i
*/
int main(int argc, char**argv)
{
   int i=1;
   int t=512;

   if (argc>1)
   {
      i=atoi(argv[1]);
      if (argc>2) 
         t=atoi(argv[2]);
   }
   else 
   {
      printf("Usage: %s i [threads] \n",argv[0]);
      return -1;
   }

   //  Problemgröße multipliziert mit i
   int size=64*i;
   
   // Datenfelder anlegen für Host
   DTYPE *vect_host, *result_h_host, *result_d_host;
   // und Device
   DTYPE *vect_dev, *result_dev;

   // Events für die Zeitmessung
   cudaEvent_t start, end;

   // Zeiten: 
   // htd: Host->Device Memcpy von A und x
   float htd_time=0.0;
   // dth: Device->Host Memcpy von y
   float dth_time=0.0;
   // kernelA
   float kernelSMRed_time=0.0;

   // DONE: Host Speicher anlegen und A und x füllen
   vect_host = (DTYPE*) malloc(size*sizeof(DTYPE));
   result_d_host = (DTYPE*) malloc(sizeof(DTYPE));
   result_h_host = (DTYPE*) malloc(sizeof(DTYPE));
   fillVect(vect_host, size);

   // DONE: CUDA Events erstellen
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // DONE: CUDA Speicher anlegen für alle Arrays (vect_dev,result_dev,y_dev)
   cudaMalloc((void**) &vect_dev, size*sizeof(DTYPE));
   cudaMalloc((void**) &result_dev, sizeof(DTYPE));
   
   // DONE: Host->Device Memcpy von A und x + Zeitmessung
   cudaEventRecord(start);
   cudaMemcpy(vect_dev, vect_host, size*sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&htd_time, start, end);

   // Konfiguration der CUDA Kernels
   dim3 threads(t);
   dim3 grid(size/threads.x);
   
   // DONE: kernelSMReduce ausführen und Zeit messen
   cudaEventRecord(start);
   kernelSMReduce<<<grid, threads>>>(vect_dev, result_dev, size);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&kernelSMRed_time, start, end);

   // DONE: Device->Host Memcpy für y_dev -> result_d_host
   cudaEventRecord(start);
   cudaMemcpy(result_d_host, result_dev, sizeof(DTYPE), cudaMemcpyDeviceToHost);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&dth_time, start, end);

   printf("GPU timing in ms: h->d: %f \nkernelSMReduce in sm %f \nd->h: %f\n",
      htd_time, kernelSMRed_time, dth_time);

   // Nutzen hier timespec um CPU Zeit zu messen
   struct timespec start_h, end_h;
   double hostSMRed_time;

   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_h);
   // DONE: A*x auf Host 
   hostSMReduce(vect_host, result_h_host, size);
   // *result_h_host = 111;
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_h);
   hostSMRed_time=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;

   printf("CPU timing in ms: kernel: SMRed: %f\n", hostSMRed_time);

   // DONE: checkResult GPU aufrufen
   if (checkResult(vect_host, result_d_host, size))
   {
      printf("GPU: jaaa es geht\n");
   }
   else
   {
      printf("GPU: musst nochmal ran :(\n");
   }

   // DONE: checkResult aufrufen
   if (checkResult(vect_host, result_h_host, size))
   {
      printf("Host: jaaa es geht\n");
   }
   else
   {
      printf("Host: musst nochmal ran :(\n");
   }

   // DONE: Speicher freigeben (Host UND Device)
   cudaFree(vect_dev);
   cudaFree(result_dev);

   free(vect_host);
   free(result_d_host);
   
   // DONE: CUDA Events zerstören
   cudaEventDestroy(start);
   cudaEventDestroy(end);

   return 0;
}
