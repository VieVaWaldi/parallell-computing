#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DTYPE float

__global__ void kernelAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   //TODO: Hier soll die GPU A*x=y berechnen
}

__global__ void kernelATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   //TODO: Hier soll die GPU A^T*x=y berechnen
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

void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   //TODO: Hier soll der Host A*x=y berechnen
}

void hostATx(DTYPE *a, DTYPE *x, DTYPE *y, int size)
{
   //TODO: Hier soll der Host A^T*x=y
}



bool checkResult(DTYPE *yh, DTYPE *yd, int size)
{
   bool res=true;
   for (int i=0;i<size;i++)
   {
      res&=(yh[i]==yd[i]);
      if (i<10) printf("%f %f\n",yh[i],yd[i]);
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
      if (argc>2) t=atoi(argv[2]);
   }
   else 
   {
      printf("Usage: %s i [threads] \n",argv[0]);
      return -1;
   }
   int size=1024*i;
   //Datenfelder anlegen für Host
   DTYPE *a_host, *yd_host, *yh_host,*x_host;
   //und Device
   DTYPE *a_dev, *y_dev,*x_dev;
   //Events für die Zeitmessung
   cudaEvent_t start,end;
   //Zeiten: 
   //htd: Host->Device Memcpy von A und x
   float htd_time=0.0;
   //dth: Device->Host Memcpy von y
   float dth_time=0.0;
   //kernelA, kernelAT
   float kernelA_time=0.0;
   float kernelAT_time=0.0;

   //TODO: Host Speicher anlegen und A und x füllen

   //TODO: CUDA Events erstellen

   //TODO: CUDA Speicher anlegen für alle Arrays (a_dev,x_dev,y_dev)
   
   //TODO: Host->Device Memcpy von A und x + Zeitmessung

   //Konfiguration der CUDA Kernels
   dim3 threads(t);
   dim3 grid(size/threads.x);
   
   //TODO: kernelAx ausführen und Zeit messen
   

   //TODO: kernelATx ausführen und Zeit messen 

   //TODO: Device->Host Memcpy für y_dev -> yd_host

   printf("GPU timing in ms: h->d: %f kernelAx: %f kernelATx: %f d->h: %f\n",htd_time,kernelA_time,kernelAT_time,dth_time);


   //Nutzen hier timespec um CPU Zeit zu messen
   struct timespec start_h,end_h;
   double hostA_time, hostAT_time;

   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
   //TODO: A*x auf Host 
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
   hostA_time=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;
   
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
   //TODO: A^T*x auf Host
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
   hostAT_time=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;

   printf("CPU timing in ms: kernel: Ax: %f  ATx: %f\n",hostA_time, hostAT_time);

   //TODO: checkResult aufrufen

   //TODO: Speicher freigeben (Host UND Device)
   
   //TODO: CUDA Events zerstören

   return 0;
}
