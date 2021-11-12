#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

#include "image.hpp"

using namespace std;

int main(){
    cout<<"hello world"<<endl;
    Image img=Image();
    img.load("img/Chateau.png");
    vector<int> rgb;
    rgb = img.getPixelRGB();    
}
