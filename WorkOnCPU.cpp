#include "WorkOnCPU.hpp"
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

            hue=60*(((b-r)/delta)+2);
        }
        else if(cMax== b){

           hue=60*(((r-g)/delta)+4);
        }
        else {

            hue=0.0;
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
   float c,x,absValue,hue,m,rTemp,gTemp,bTemp;
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
    for(int i=0;i<256;i++){
    }
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


