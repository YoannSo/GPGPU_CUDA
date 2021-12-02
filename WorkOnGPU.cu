
#include "WorkOnGPU.hpp"

int* workOnGpu(vector<int> rgb){
    float totalTime=0;
    int HSVsize=rgb.size()/3;
    vector<float> vectorh=vector<float>();
    vector<float> vectors=vector<float>();
    vector<float> vectorv=vector<float>();
    int nbPixels=rgb.size()/3+1;
    int nbBlocks;
    float* h=new float[HSVsize];
    float* s=new float[HSVsize];
    float* v=new float[HSVsize];
    vector<int> newRGBVector=vector<int>();
    int* dev_histo=new int[256];
    int* dev_repartition=new int[256];
    ChronoGPU chrGPU;

    cout << "============================================"	<< endl;
	cout << "         Parallel version on GPU          "	<< endl;
	cout << "============================================"	<< endl;
    
    float* dev_h=new float[HSVsize];
    float* dev_s=new float[HSVsize];
    float* dev_v=new float[HSVsize];
    int * dev_rgb;
    int* newRGB = new int[rgb.size()];
    int* result = new int[rgb.size()];
    float* newV;


    HANDLE_ERROR(cudaMalloc(&newRGB, rgb.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_rgb, rgb.size() * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_h, HSVsize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_s, HSVsize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_v, HSVsize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_histo, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_repartition, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&newV, HSVsize * sizeof(float)));
    
    HANDLE_ERROR(cudaMemcpy(dev_rgb, rgb.data(), rgb.size() * sizeof(int), cudaMemcpyHostToDevice));
    nbBlocks=nbPixels/(1024)+1;
    int minGrid;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid,&blockSize,RGBtoHSV_GPU,0,nbPixels);
    int grid=(nbPixels+blockSize-1)/blockSize;
        cout<<minGrid<<" "<<blockSize<<" "<<grid<<" "<<nbBlocks<<endl;

    chrGPU.start();
    RGBtoHSV_GPU<<<nbBlocks,1024>>>(dev_rgb,dev_h,dev_s,dev_v,HSVsize); 
    chrGPU.stop();  
   
    cout << "-> RGB to HSV done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();    
    int * temp=new int[256];
    for(int i=0;i<256;i++){
        temp[i]=0;
    }
    HANDLE_ERROR(cudaMemcpy(dev_histo, temp, 256*sizeof(int), cudaMemcpyHostToDevice));
    nbBlocks=nbPixels/(1024*3)+1;
    chrGPU.start();
    computeHistogram_GPU_sharedMemoryVersion <<<32,32 >>>(dev_v, dev_histo,HSVsize);
    //computeHistogram_GPU <<<nbBlocks,1024 >>>(dev_v, dev_histo,HSVsize);
    chrGPU.stop();


    cout << "-> Compute histogram done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();


    chrGPU.start();
    repartition_GPU<<<1, 256 >>>(dev_histo,dev_repartition,256);
    chrGPU.stop();
    cout << "-> Compute Repartition done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();

   
    
       
    chrGPU.start();
    egalisation_GPU <<<nbBlocks, 256 >>>(dev_repartition,dev_v,newV,HSVsize);
    chrGPU.stop();


    cout << "-> Compute Egalisation done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();
    nbBlocks=nbPixels/(1024*3)+1;
    chrGPU.start();
    HSVtoRGB_GPU<<<nbBlocks,1024>>>(dev_h,dev_s,newV,newRGB,HSVsize);
    chrGPU.stop();
    float* test = new float[HSVsize];
    HANDLE_ERROR(cudaMemcpy(test, newV, HSVsize*sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(result, newRGB, rgb.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cout << "-> HSV to FINAL RGB done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();
    cout << "-> ALL GPU DONE total time : " << fixed << setprecision(2) << totalTime<< " ms" << endl << endl;
    

    HANDLE_ERROR(cudaFree(newV));
    HANDLE_ERROR(cudaFree(dev_histo));
    HANDLE_ERROR(cudaFree(dev_repartition));
    HANDLE_ERROR(cudaFree(dev_rgb));
    HANDLE_ERROR(cudaFree(dev_h));
    HANDLE_ERROR(cudaFree(dev_s));
    HANDLE_ERROR(cudaFree(dev_v));
    HANDLE_ERROR(cudaFree(newRGB));
    
    return result;
}


__global__ void RGBtoHSV_GPU( int* dev_rgb,float* dev_h,float* dev_s,float* dev_v,int rgbSize){// 1threads =rgb
    int tid =blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < rgbSize) {
       float r,g,b;
       float cMax,cMin,delta;
        r=(float) dev_rgb[tid*3]/255;
        g=(float) dev_rgb[tid*3+1]/255;
        b=(float) dev_rgb[tid*3+2]/255;
        

        cMax=max(r,g);
        cMax=max(cMax,b);
        cMin=min(r,g);
        cMin=min(cMin,b);
        delta = cMax-cMin;
    
        // Calcul de H
        float hue=0;
        if(cMax == r){
            float left=((g-b)/delta);
            int quot=left/6;
            float mod=left-quot*6;
            hue=60 * mod;
        }else if (cMax == g){
            hue=60*(((b-r)/delta)+2);//Calcul a faire
        }
        else if(cMax== b){
           hue=60*(((r-g)/delta)+4);
        }
        else {
            hue=0.0;//Calcul a faire
        }
        if(hue<0){
            hue+=360;
        }
        dev_h[tid]=hue;
        // Calcul de S
        if (cMax>0.0){
            dev_s[tid]=delta/cMax;
        } 
        else {
            dev_s[tid]=0;
        }
        // Calcul de V
        dev_v[tid]=cMax;
        tid += gridDim.x * blockDim.x;
    
   }
}
__global__ void HSVtoRGB_GPU(float* h,float* s,float* newV,int* rgb,int HSVsize){
int tid =blockIdx.x * blockDim.x + threadIdx.x;
   while (tid < HSVsize) {
       
    float c,x,hue,m,rTemp,gTemp,bTemp;
    hue=h[tid];
    c= newV[tid]*s[tid];
    int quot=(int)((hue/60)/2);
    float fmod=hue/60-(quot*2.0);
    fmod-=1;
    if(fmod<0)
        fmod=-fmod;    
       x=c*(1-fmod);
       m= newV[tid]-c;
    if(hue>=0 && hue<60){
        rTemp=c;
        gTemp=x;
        bTemp=0;
    }
    else if(hue>=60 && hue<120){
        rTemp=x;
        gTemp=c;
        bTemp=0;
    }
    else if(hue>=120 && hue<180){
        rTemp=0;
        gTemp=c;
        bTemp=x;     
    }
    else if(hue>=180 && hue<240){
        rTemp=0;
        gTemp=x;
        bTemp=c;
    }
    else if(hue>=240 && hue<300){
        rTemp=x;
        gTemp=0;
        bTemp=c;
    }
    else{ //300 360
        rTemp=c;
        gTemp=0;
        bTemp=x;
    }
    rTemp=(rTemp+m)*255;
    gTemp=(gTemp+m)*255;
    bTemp=(bTemp+m)*255;
    
    rgb[tid*3]=int(rTemp);
    rgb[tid*3+1]=int(gTemp);
    rgb[tid*3+2]=int(bTemp);
        
    tid += gridDim.x * blockDim.x;
   }


}
__global__
void computeHistogram_GPU(const float* dev_v , int* dev_histo, const int v_size){
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
     while (tid < v_size) {
         atomicAdd(&dev_histo[int(dev_v[tid] * 255)],1);
         tid += gridDim.x * blockDim.x;
     }
}

__global__
void computeHistogram_GPU_sharedMemoryVersion(const float* dev_v, int* dev_histo, const int v_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int histo[256];
    if (tid < 256) {
        histo[tid] = 0;
        dev_histo[tid] = 0;
    }
    __syncthreads();
    while (tid < v_size) {
        atomicAdd(&histo[int(dev_v[tid] * 255)], 1);
        tid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 256; i++) {
            atomicAdd(&dev_histo[i], histo[i]);
        }
    }
}



__global__
void repartition_GPU(int* dev_histo,int* dev_repartition, int size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val;
    
    if (tid < size) {
        dev_repartition[tid] = dev_histo[tid];
        val = dev_histo[tid];
    }
    __syncthreads();
    for (int i = tid+1; i < size; i++) {
        atomicAdd(&dev_repartition[i],val);
    }
    
   
   
}

__global__
void egalisation_GPU(const int* dev_repartition, float* dev_v,  float* newV ,const int v_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < v_size) {
        newV[tid] = (255.0 / (256.0 * v_size)) * dev_repartition[int(dev_v[tid] * 255)];
        tid += gridDim.x * blockDim.x;
    }
    
}