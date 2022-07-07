#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DTYPE float

/**
   Matrix Vector - Gaxpy
   
   => y = y + Ax
   
   - A = m, n matrix
   - x = n vector
   - y = m vector
*/

/********************************* Kernel 1 - Ax simple *********************************/

__global__ void kernel_Ax_simple(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < size)
   {
      int sum_y = 0;
      int offset_a = i*size;
      for( int j = 0; j < size; j++)
      {
         sum_y += a[offset_a+j] * x[j];
      }
      y[i] = sum_y;  
   }
}

/********************************* Kernel 2 - SM *********************************/
/* Sorry das habe ich nicht hin bekommen :( Dieser Kernel geht nicht .... */

__global__ void kernel_Ax_SM(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;

   int idx = i + j * size;
   int smIdx = threadIdx.x + threadIdx.y * blockDim.x;

   extern __shared__ DTYPE sm[4048];

   if (i < size && j < size)
   {
      sm[smIdx] = a[idx] * x[i];
      __syncthreads();

      for (int k = blockDim.x/2; k>0; k/=2)
      {
         if (threadIdx.x < k)
         {
            sm[smIdx] += sm[smIdx + k];
         }
         __syncthreads();
      }

      if (threadIdx.x == 0)
      {
         y[j] += sm[threadIdx.y * blockDim.x];
      }
   }
}

/********************************* Kernel 3 - SMAtomicEnd *********************************/

__global__ void kernel_Ax_SMAtomicEnd(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;

   int idx = i + j * size;
   int smIdx = threadIdx.x + threadIdx.y * blockDim.x;

   // printf("GPU: j = %d, i = %d, idx = %d, simIdx = %d\n", j, i, idx, smIdx);

   extern __shared__ DTYPE sm[4048];

   if (i < size && j < size)
   {
      sm[smIdx] = a[idx] * x[i];
      __syncthreads();

      for (int k = blockDim.x/2; k>0; k/=2)
      {
         if (threadIdx.x < k)
         {
            sm[smIdx] += sm[smIdx + k];
         }
         __syncthreads();
      }

      if (threadIdx.x == 0)
      {
         atomicAdd(&y[j], sm[threadIdx.y * blockDim.x]);
      }
   }
}

/********************************* Kernel 4 - AtomicOnly *********************************/

__global__ void kernel_Ax_AtomicOnly(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;

   int idx = i + j * size;

   if (i<size && j<size)
   {
      atomicAdd(&y[j], a[idx] * x[i]);
   }
}

/********************************* Host *********************************/

void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   for( int i = 0; i < size; i++)      // y und a.y durchgehen
   {
      int sum_y = 0;
      int offset_a = i*size;           // offset++ nach dem size elemente durchgelaufen sind
      for( int j = 0; j < size; j++)   // a.x und x durchgehen
      {
         sum_y += a[offset_a+j] * x[j];
      }
      y[i] = sum_y;  
   }
}

/********************************* Helper *********************************/

void fillA(DTYPE *a, int size)
{
   for (int i=0;i<size*size;i++)
      a[i]=1.0;
}

void fillX(DTYPE *x, int size)
{
   for (int i=0;i<size;i++)
      x[i]= (DTYPE)(i+1);
}

void checkResult(DTYPE *yh, DTYPE *yd, int size, char* kernelName)
{
   bool res = true;

   for (int i = 0; i < size; i++)
   {
      res &= ( yh[i] == yd[i] );
      // if (i<3) 
      //    printf("%f %f\n",yh[i],yd[i]);
   }

   if (res)
   {
      printf("%s GEHT (((: \n\n", kernelName);
   }
   else
   {
      printf("%s GEHT NICHT :( ...\n\n", kernelName);
   }
}

void resetResultVector(DTYPE *y_dev, DTYPE *yd_host, int size) 
{
   for (int i=0; i<size; i++)
   {
      yd_host[i]=0;
   }
   cudaMemcpy(y_dev, yd_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaDeviceSynchronize();
}

void checkCudaError()
{
   // Make the host block until the device is finished with foo
   cudaDeviceSynchronize();

   // Check for error
   cudaError_t error = cudaGetLastError();
   if( error != cudaSuccess )
   {
      printf("<<< CUDA error: %s >>>\n\n", cudaGetErrorString(error));
   }
}

/**
   Main Routine: 
   Input: i
   Berechnet A*x=y auf der GPU wobei A eine Größe von R^{n x n} hat, mit
   n=1024*i
*/

// Events für die Zeitmessung
cudaEvent_t start, end;

int main(int argc, char**argv)
{
   int i = 1, cacheConf = 0;
   if (argc > 1)
   {
      i=atoi(argv[1]);
      cacheConf=atoi(argv[2]);

      if (cacheConf == 1) {
         printf("Using cudaFuncCachePreferL1\n");
         cudaFuncSetCacheConfig( kernel_Ax_simple, cudaFuncCachePreferL1 );
         cudaFuncSetCacheConfig( kernel_Ax_SMAtomicEnd, cudaFuncCachePreferL1 );
         cudaFuncSetCacheConfig( kernel_Ax_SMAtomicEnd, cudaFuncCachePreferL1 );
      }
      else if (cacheConf == 2) {
         printf("Using cudaFuncCachePreferShared\n");
         cudaFuncSetCacheConfig( kernel_Ax_simple, cudaFuncCachePreferShared );
         cudaFuncSetCacheConfig( kernel_Ax_SMAtomicEnd, cudaFuncCachePreferShared );
         cudaFuncSetCacheConfig( kernel_Ax_SMAtomicEnd, cudaFuncCachePreferShared );
      }
   }
   else 
   {
      printf("Usage: %s [problemMultiplier] [chacheConf]\n", argv[0]);
      return -1;
   }

   // Problemgröße 1024 multipliziert mit i
   int prob = 1024;
   int size = prob * i;
   printf("Problem size is %d * %d = %d\n\n", prob, i, size);
   
   // Datenfelder anlegen für Host und Device
   DTYPE *a_host, *x_host, *yh_host, *yd_host;
   DTYPE *a_dev, *x_dev, *y_dev;

   // Host Speicher anlegen
   a_host = (DTYPE*) malloc( size * size * sizeof(DTYPE) );
   x_host = (DTYPE*) malloc( size * sizeof(DTYPE) );
   yh_host = (DTYPE*) malloc( size * sizeof(DTYPE) );
   yd_host = (DTYPE*) malloc( size * sizeof(DTYPE) );

   // Host arrays füllen
   fillA( a_host, size );
   fillX( x_host, size );

   // CUDA Speicher anlegen für alle Arrays
   cudaMalloc( (void**) &a_dev, size * size * sizeof(DTYPE) );
   cudaMalloc( (void**) &x_dev, size * sizeof(DTYPE) );
   cudaMalloc( (void**) &y_dev, size * sizeof(DTYPE) );

   // Host execution //
   struct timespec start_h, end_h;
   double hostAx_time;

   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
   hostAx(a_host, x_host, yh_host, size);
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
   hostAx_time = (double) ( (end_h.tv_nsec+end_h.tv_sec * 1E9) - (start_h.tv_nsec+start_h.tv_sec * 1E9) ) / 1E6;

   printf("HOST hostAx timing in ms %f\n\n", hostAx_time);

   // Zeiten: 
   // htd: Host->Device Memcpy von A und x
   float htd_time=0.0;
   // dth: Device->Host Memcpy von y
   float dth_time=0.0;

   // CUDA Events erstellen
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // Host -> Device, load all data to GPU
   cudaEventRecord(start);
   cudaMemcpy(a_dev, a_host, size * size * sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&htd_time, start, end);

   printf("GPU h -> d in ms %f\n\n", htd_time);

   /********************************* Kernel 1 Ax - simple */

   dim3 thread( 512 );
   dim3 grid( size / thread.x);
   
   float Ax_simple_time = 0.0;

   // Kernel execution and time measurement
   cudaEventRecord(start);
   
   kernel_Ax_simple<<<grid, thread>>>(a_dev, x_dev, y_dev, size);
   
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&Ax_simple_time, start, end);

   // Device -> Host Memcpy
   cudaEventRecord(start);
   cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&dth_time, start, end);

   printf("GPU timing 1. KernelAx simple in ms:\nkernel_Ax_simple: %10f\nd -> h: %20f\n",
      Ax_simple_time, dth_time);

   checkResult(yh_host, yd_host, size, "kernel_Ax_simple");
   resetResultVector( y_dev, yd_host, size );

   /********************************* Kernel 2 Ax - kernel_Ax_SM */

   /* Dieser Kernel geht leider nicht und ich hab nur noch eine Stunde. Entschuldigung :( */

   // float Ax_SM_time = 0.0;

   // // Kernel execution and time measurement
   // cudaEventRecord(start);

   // for (int i=0; i<1; i++) {
   //    kernel_Ax_SM<<<grids, threads>>>(a_dev, x_dev, y_dev, size);
   //    cudaDeviceSynchronize();
   // }
   
   // cudaEventRecord(end);
   // cudaEventSynchronize(end);
   // cudaEventElapsedTime(&Ax_SM_time, start, end);

   // // Device -> Host Memcpy
   // cudaEventRecord(start);
   // cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
   // cudaEventRecord(end);
   // cudaEventSynchronize(end);
   // cudaEventElapsedTime(&dth_time, start, end);

   // printf("GPU timing 2. KernelAx SM in ms:\nkernel_Ax_SM: %10f\nd -> h: %25f\n",
   //    Ax_SM_time, dth_time);

   // checkCudaError();
   // checkResult(yh_host, yd_host, size, "kernel_Ax_SM");
   // resetResultVector( y_dev, yd_host, size );

   /********************************* Kernel 3 Ax - SMAtomicEnd */

   // Konfiguration der CUDA Kernels
   dim3 threads( 32, 32 ); // = 1024
   dim3 grids( size / threads.x, size / threads.y );
   // dim3 grid( 32, 32 );
   
   float Ax_SMAtomicEnd_time = 0.0;

   // Kernel execution and time measurement
   cudaEventRecord(start);

   kernel_Ax_SMAtomicEnd<<<grids, threads>>>(a_dev, x_dev, y_dev, size);
   
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&Ax_SMAtomicEnd_time, start, end);

   // Device -> Host Memcpy
   cudaEventRecord(start);
   cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&dth_time, start, end);

   printf("GPU timing 3. KernelAx SMAtomicEnd in ms:\nkernel_Ax_SMAtomicEnd: %10f\nd -> h: %25f\n",
      Ax_SMAtomicEnd_time, dth_time);

   checkCudaError();
   checkResult(yh_host, yd_host, size, "kernel_Ax_SMAtomicEnd");
   resetResultVector( y_dev, yd_host, size );

   /********************************* Kernel 4 Ax - Atomic Only */
   
   float Ax_AtomicOnly_time = 0.0;

   // Kernel execution and time measurement
   cudaEventRecord(start);

   kernel_Ax_AtomicOnly<<<grids, threads>>>(a_dev, x_dev, y_dev, size);
   
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&Ax_AtomicOnly_time, start, end);

   // Device -> Host Memcpy
   cudaEventRecord(start);
   cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&dth_time, start, end);

   printf("GPU timing 4. KernelAx AtomicOnly in ms:\nkernel_Ax_AtomicOnly: %10f\nd -> h: %24f\n",
      Ax_AtomicOnly_time, dth_time);

   checkCudaError();
   checkResult(yh_host, yd_host, size, "kernel_Ax_AtomicOnly");

   /**************************************************/

   // Speicher freigeben (Host UND Device)
   cudaFree(a_dev);
   cudaFree(x_dev);
   cudaFree(y_dev);
   free(a_host);
   free(x_host);
   free(yh_host);
   free(yd_host);
   
   // CUDA Events zerstören
   cudaEventDestroy(start);
   cudaEventDestroy(end);

   return 0;
}
