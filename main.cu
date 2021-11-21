#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>    // std::max
#include "image.hpp"
#include <math.h>
#include <cstdlib>
#include <iomanip>    
#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"
#define PI 3.141592
using namespace std;

vector<int> workOnCpu(vector<int> rgb);
void RGBtoHSV_CPU (int* rgb, vector<float>& h, vector<float>& s, vector<float>& v,int size);
void HSVtoRGB_CPU(float* h,float *s,float*v,int size,vector<int>&rgb);
int* histo_CPU(float*v,int size);
int* repartition_CPU(int* histo,int size);
vector<float> egalisation_CPU(int* repartition, vector<float> v);

vector<int> workOnGpu(vector<int> rgb);
void RGBtoHSV_GPU(vector<int> rgb,vector<float>&h,vector<float>&s,vector<float>&v);
int* computeHistogram_GPU(vector<float>v);
int* repartition_GPU(int* histo,int size);
vector<float> egalisation_GPU(int* repartition,vector<float>v);
void HSVtoRGB_GPU(vector<float> h,vector<float>s,vector<float>v,vector<int>&rgb);


int main(){
    cout<<"hello world"<<endl;
    Image img=Image();
    img.load("img/Chateau.png");
    vector<int> newRGB_CPU=workOnCpu(img.getPixelRGB());
    img.setRGB(newRGB_CPU);
    img.setPixels();
    img.save("img/ChateauCPU.png");

    img.load("img/Chateau.png");
    vector<int> newRGB_GPU=workOnGpu(img.getPixelRGB());
    img.setRGB(newRGB_GPU);
    img.setPixels();
    img.save("img/ChateauGPU.png");
       
}


///////////////////////// CPU PART /////////////////////////////////////////
vector<int> workOnCpu(vector<int> rgb){
    float totalTime=0;
    vector<float> h=vector<float>();
    vector<float> s=vector<float>();
    vector<float> v=vector<float>();
    ChronoCPU chrCPU;

    cout << "============================================"	<< endl;
	cout << "         Sequential version on CPU          "	<< endl;
	cout << "============================================"	<< endl;
    chrCPU.start();
    RGBtoHSV_CPU(rgb.data(),h,s,v,rgb.size());
    chrCPU.stop();
    totalTime += chrCPU.elapsedTime();
    cout << "-> RGB to HSV done : " << fixed << setprecision(2) << chrCPU.elapsedTime() << " ms" << endl << endl;
    chrCPU.start();

    int* histogram=new int[256];
    histogram=histo_CPU(v.data(),v.size());
    
    chrCPU.stop();
    cout << "-> Compute histogram done : " << fixed << setprecision(2) << chrCPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrCPU.elapsedTime();
    chrCPU.start();

    int* repartition=new int[256];
    repartition=repartition_CPU(histogram,256);

    chrCPU.stop();
    cout << "-> Compute Repartition done : " << fixed << setprecision(2) << chrCPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrCPU.elapsedTime();    
    chrCPU.start();

    vector<float> newV=vector<float>();
    newV = egalisation_CPU(repartition, v); 

    chrCPU.stop();
    cout << "-> Compute Egalisation histograme done : " << fixed << setprecision(2) << chrCPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrCPU.elapsedTime();
    chrCPU.start();

    vector<int> newRGB;
    HSVtoRGB_CPU(h.data(), s.data(), newV.data(), h.size(), newRGB);

    chrCPU.stop();
    cout << "-> HSV to RGB done : " << fixed << setprecision(2) << chrCPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrCPU.elapsedTime();
    cout << "-> ALL CPU DONE total time : " << fixed << setprecision(2) << totalTime<< " ms" << endl << endl;
    return newRGB;
}
void RGBtoHSV_CPU (int* rgb, vector<float>& h, vector<float>& s, vector<float>& v,int size){
    float r,g,b;
    float cMax,cMin,delta;

    for (int i=0;i<size;i+=3){
        r=(float) rgb[i]/255;
        g=(float) rgb[i+1]/255;
        b=(float) rgb[i+2]/255;
        cMax=max(r,g);
        cMax=max(cMax,b);
        cMin=min(r,g);
        cMin=min(cMin,b);
        delta = cMax-cMin;
        // Calcul de H
        float hue;
        if(cMax == r){
            hue=60 * (fmod(((g - b) / delta), 6));
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
        h.push_back(hue);
        // Calcul de S
        if (cMax>0.0){

            s.push_back(delta/cMax);
        } else {

            s.push_back(0);
        }
        // Calcul de V
        v.push_back(cMax);
      

    }

}
void HSVtoRGB_CPU(float* h,float *s,float*v,int size,vector<int>&rgb){
   float c,x,absValue,m,hue,rTemp,gTemp,bTemp;
   for(int i=0;i<size;i++){
       hue=h[i];
       c=v[i]*s[i];
       absValue=fabs(fmod(hue/60.0,2)-1);
       x=c*(1-absValue);
       m=v[i]-c;
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
    

    rgb.push_back(int(rTemp));
    rgb.push_back(int(gTemp));
    rgb.push_back(int(bTemp));

   }
}

int* histo_CPU(float*v,int size){
        int* histogram=new int[256]; // de 0 a 255 valeur de V
        for(int i=0;i<256;i++){
            histogram[i]=0;
        }
        
        for(int i=0;i<size;i++){
            histogram[int(v[i]*255)]++;        
        }
        return histogram;
}
int* repartition_CPU(int* histo,int size){
        int* repartition=new int[256]; // de 0 a 255 valeur de V
        int histoValue;
        for(int i=0;i<size;i++){
            histoValue=0;
            for(int j=0;j<i;j++){
                histoValue+=histo[j];
            }
            histoValue+=histo[i];
            repartition[i]=histoValue;
        
        }
       
        return repartition;
}

vector<float> egalisation_CPU(int* repartition, vector<float> v) {

    float* egalisation = new float[256];
    vector<float> result= vector<float>(v.size());
    
    for(int i=0;i<result.size();i++){
        result[i]=0;
    }
    for (int i = 0; i < 256; i++) {
        egalisation[i] = (255.0 / (256.0 * v.size())) *repartition[i];
    }
    for (int i = 0; i < v.size(); i++) {
       result[i]=egalisation[(int(v[i] * 255))];
    }
    return result;
}


//////////////////////////////////////////////// GPU PART //////////////////////////////////////////////////
vector<int> workOnGpu(vector<int> rgb){
    float totalTime=0;
    vector<float> h=vector<float>();
    vector<float> s=vector<float>();
    vector<float> v=vector<float>();
    vector<int> newRGB=vector<int>();
    int* histo=new int[256];
    int* repartition=new int[256];
    vector<float>newV=vector<float>();
    ChronoGPU chrGPU;

    cout << "============================================"	<< endl;
	cout << "         Parallel version on GPU          "	<< endl;
	cout << "============================================"	<< endl;
    
    chrGPU.start();
    RGBtoHSV_GPU(rgb,h,s,v);
    chrGPU.stop();
    cout << "-> RGB to HSV done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();    
  
    chrGPU.start();
    histo=computeHistogram_GPU(v);
    chrGPU.stop();
    cout << "-> Compute histogram done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();

    chrGPU.start();
    repartition=repartition_GPU(histo,256);
    chrGPU.stop();
    cout << "-> Compute Repartition done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();

    chrGPU.start();
    newV=egalisation_GPU(histo,v);
    chrGPU.stop();
    cout << "-> Compute Egalisation done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();

    chrGPU.start();
    HSVtoRGB_GPU(h,s,v,newRGB);
    chrGPU.stop();
    cout << "-> HSV to FINAL RGB done : " << fixed << setprecision(2) << chrGPU.elapsedTime() << " ms" << endl << endl;
    totalTime += chrGPU.elapsedTime();
    cout << "-> ALL CPU DONE total time : " << fixed << setprecision(2) << totalTime<< " ms" << endl << endl;
    return newRGB;
}
void RGBtoHSV_GPU(vector<int> rgb,vector<float>&h,vector<float>&s,vector<float>&v){

}
int* computeHistogram_GPU(vector<float>v){

}
int* repartition_GPU(int* histo,int size){

}
vector<float> egalisation_GPU(int* repartition,vector<float>v){

}
void HSVtoRGB_GPU(vector<float> h,vector<float>s,vector<float>v,vector<int>&rgb){

}