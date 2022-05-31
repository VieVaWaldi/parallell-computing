#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DTYPE float

/**
   Matrix Vector - Gaxpy
   
   - y = y + Ax
   
   - A = m, n matrix
   - x = n vector
   - y = m vector
   
   - A ist hier auch nur ein 1 Dim Array, ABER
   mit der Länge size*size was es zu einem vector macht.
   - A indexe werden also mit i*size (=ydim) + j (=xdim) berechnet.
   - x hat dann obviously nur die Länge size und y ebenfalls weil symmetrie. 

   Matrix Vector - Gaxpy Transponiert

   - Bei A einfach anstatt die Zeilen durch zu gehen
   durch die Columns gehen, also nur Indizes aendern.
   - Das simuliert den Zugriff der threads auf verschobene
   cacheline
*/

//DONE: Hier soll die GPU A*x=y berechnen
__global__ void kernelAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   /** 
      Kernel Denken
      
      - Immer noch nur 1 dimensionales denken (weil ohne sync).
      - Jeder Thread i bekommt die Kontrolle über ein Element von y.
      - i summiert in der Zeile i von A die Spalten mit allen Elementen von x.
      - Im Vergleich zum Host wird die oebere Schleife durch Threads ersetzt.
   */

   int tid = threadIdx.x;
   int bid = blockIdx.x;
   int bdim = blockDim.x;

   int i = tid+bid*bdim;

   if (i < size)                       // Die Threads sind eine Ele in y und die A.Zeilen
   {
      int sum_y = 0;
      int offset_a = i*size;
      for( int j = 0; j < size; j++)   // nur a.x und x durchgehen
      {
         sum_y += a[offset_a+j] * x[j];
      }
      y[i] = sum_y;  
   }
}

//DONE: Hier soll die GPU A^T*x=y berechnen
__global__ void kernelATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   /** 
      A Transpose Denken
      
      - Der offset wird nicht mit i bestimmt, und 
   */

   int tid = threadIdx.x;
   int bid = blockIdx.x;
   int bdim = blockDim.x;

   int i = tid+bid*bdim;

   if (i < size)                       
   {
      int sum_y = 0;
      for( int j = 0; j < size; j++)   
      {
         int offset_a = j*size;
         sum_y += a[offset_a+i] * x[j];
      }
      y[i] = sum_y;  
   }
}



//A mit Werten füllen (hier einfach 1en)
void fillA(DTYPE *a, int size)
{
   for (int i=0;i<size*size;i++)
      a[i]=1.0;
}

//X mit Werten füllen 
void fillX(DTYPE *x, int size)
{
   for (int i=0;i<size;i++)
      x[i]= (DTYPE)(i+1);
}

//DONE: HOST berechnet Gaxpy
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

//TODO: Hier soll der Host A^T*x=y
void hostATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   // Für A Transpose müssen nur die Indizes gewechselt werden

   for( int i = 0; i < size; i++)
   {
      int sum_y = 0;
      for( int j = 0; j < size; j++)
      {
         int offset_a = j*size;           // offset springt jeden schritt um size elemente
         sum_y += a[offset_a+i] * x[j];   // summiert von a werden die zeilen
      }
      y[i] = sum_y;  
   }
}


bool checkResult(DTYPE *yh, DTYPE *yd, int size)
{
   bool res=true;
   for (int i=0;i<size;i++)
   {
      res&=(yh[i]==yd[i]);
      if (i<10) 
         printf("%f %f\n",yh[i],yd[i]);
   }
   return res;
}

/*
   Main Routine: 
   Input: i,[threads]
   Berechnet A*x=y auf der GPU wobei A eine Größe von R^{n x n} hat, mit
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

   // Problemgröße multipliziert mit i
   int size=1024*i;
   
   //Datenfelder anlegen für Host
   DTYPE *a_host, *yd_host, *yh_host, *x_host;
   //und Device
   DTYPE *a_dev, *y_dev, *x_dev;

   //Events für die Zeitmessung
   cudaEvent_t start,end;

   //Zeiten: 
   //htd: Host->Device Memcpy von A und x
   float htd_time=0.0;
   //dth: Device->Host Memcpy von y
   float dth_time=0.0;
   //kernelA
   float kernelA_time=0.0;
   //kernelAT
   float kernelAT_time=0.0;

   //DONE: Host Speicher anlegen und A und x füllen
   a_host = (DTYPE*)malloc(size*size*sizeof(DTYPE));
   x_host = (DTYPE*)malloc(size*sizeof(DTYPE));
   yh_host = (DTYPE*)malloc(size*sizeof(DTYPE));
   yd_host = (DTYPE*)malloc(size*sizeof(DTYPE));
   fillA(a_host, size);
   fillX(x_host, size);

   //DONE: CUDA Events erstellen
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   //DONE: CUDA Speicher anlegen für alle Arrays (a_dev,x_dev,y_dev)
   cudaMalloc((void**)&a_dev,size*size*sizeof(DTYPE));
   cudaMalloc((void**)&x_dev,size*sizeof(DTYPE));
   cudaMalloc((void**)&y_dev,size*sizeof(DTYPE));
   
   //DONE: Host->Device Memcpy von A und x + Zeitmessung
   cudaEventRecord(start);
   cudaMemcpy(a_dev,a_host, size*size*sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaMemcpy(x_dev,x_host, size*sizeof(DTYPE), cudaMemcpyHostToDevice);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&htd_time, start, end);

   //Konfiguration der CUDA Kernels
   dim3 threads(t);
   dim3 grid(size/threads.x);
   
   //DONE: kernelAx ausführen und Zeit messen
   cudaEventRecord(start);
   kernelAx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&kernelA_time, start, end);

   //DONE: kernelATx ausführen und Zeit messen 
   cudaEventRecord(start);
   kernelATx<<<grid, threads>>>(a_dev, x_dev, y_dev, size);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&kernelAT_time, start, end);

   //DONE: Device->Host Memcpy für y_dev -> yd_host
   cudaEventRecord(start);
   cudaMemcpy(yd_host, y_dev, size*sizeof(DTYPE), cudaMemcpyDeviceToHost);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&dth_time, start, end);

   printf("GPU timing in ms: h->d: %f kernelAx: %f kernelATx: %f d->h: %f\n",
      htd_time, kernelA_time, kernelAT_time, dth_time);

   //Nutzen hier timespec um CPU Zeit zu messen
   struct timespec start_h,end_h;
   double hostA_time, hostAT_time;

   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
   //DONE: A*x auf Host 
   hostAx(a_host, x_host, yh_host, size);
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
   hostA_time=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;
   
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
   //DONE: A^T*x auf Host
   hostATx(a_host, x_host, yh_host, size);
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
   hostAT_time=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;

   printf("CPU timing in ms: kernel: Ax: %f  ATx: %f\n",hostA_time, hostAT_time);

   //DONE: checkResult aufrufen
   if (checkResult(yh_host, yd_host, size))
   {
      printf("jaaa es geht\n");
   }
   else
   {
      printf("musst nochmal ran :(\n");
   }

   //DONE: Speicher freigeben (Host UND Device)
   cudaFree(a_dev);
   cudaFree(x_dev);
   cudaFree(y_dev);
   free(a_host);
   free(x_host);
   free(yh_host);
   free(yd_host);
   
   //DONE: CUDA Events zerstören
   cudaEventDestroy(start);
   cudaEventDestroy(end);

   return 0;
}
