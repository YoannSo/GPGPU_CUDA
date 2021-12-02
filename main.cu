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
#include "utils/common.hpp"
#include "WorkOnGPU.hpp"
#include "WorkOnCPU.hpp"

using namespace std;





int main(){
    int userInput=-1;
    Image imgForCPU=Image();
    Image imgForGPU=Image();
    while(true){
        cout << "Tapez un nombre en fonction de votre utilisation"<<endl;
        cout << "1: Utilisation du programme avec l'image du sujet"<<endl;
        cout << "2: Utilisation du programme avec une image de taille normale"<<endl;
        cout << "3: Utilisation du programme avec une image en 4k"<<endl;
        cout << "0: quittez le programme"<<endl;
        cin>>userInput;
        if(userInput==1){
            imgForCPU.load("img/Chateau.png");
            imgForGPU.load("img/Chateau.png");
            break;
        }
        else if(userInput==2){
            imgForCPU.load("img/voiture.png");
            imgForGPU.load("img/voiture.png");
            break;
        }         
        else if(userInput==3){
            imgForCPU.load("img/japan4k.png");
            imgForGPU.load("img/japan4k.png");
            break;
        }
        else if(userInput==0){
            return 0;
        }
        
        continue;
        }
        
    
    cout<<"Taille de l'image, width:"<<imgForCPU._width<<" heigth:"<<imgForCPU._height<<endl;

    vector<int> newRGB_CPU=workOnCpu(imgForCPU.getPixelRGB());
    int* newRGB_GPU=workOnGpu(imgForGPU.getPixelRGB());


    imgForCPU.setRGB(newRGB_CPU);
    imgForCPU.setPixels();
    imgForCPU.save("img/resultCPU.png");
    vector<int> newRGBVector_GPU = vector<int>();
    

    for(int i=0;i<newRGB_CPU.size();i++){
        newRGBVector_GPU.push_back(newRGB_GPU[i]);
    }
        
    imgForGPU.setRGB(newRGBVector_GPU);
    imgForGPU.setPixels();
    imgForGPU.save("img/resultGPU.png");
}



