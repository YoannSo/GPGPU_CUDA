#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

#include "image.hpp"

using namespace std;

void RGBtoHSV (vector<int> rgb, vector<float> h, vector<float>s, vector<float>v){
    float r,g,b,cMax,cMin,delta;
    for (int i=0;i<rgb.size();i++){
        r= rgb[i*3]/255;
        g= rgb[i*3+1]/255;
        b= rgb[i*3+2]/255;
        cMax=max(r,g,b);
        cMin=min(r,g,b);
        delta = cMax-cMin;
        // Calcul de H
        if(cMax == r){
            h.push_back(0.0); //Calcul a faire
        }else if (cMax == g){
            h.push_back(0.0);//Calcul a faire
        }else {
            h.push_back(0.0);//Calcul a faire
        }
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


int main(){
    cout<<"hello world"<<endl;
    Image img=Image();
    img.load("img/Chateau.png");
    vector<int> rgb;
    rgb = img.getPixelRGB();    
    
    
    
}
