#ifndef __WORK_ON_CPU_HPP
#define __WORK_ON_CPU_HPP

#include "utils/common.hpp"
#include "utils/chronoCPU.hpp"
#include <iomanip> 
#include <iostream>
#include <vector>
using namespace std;
vector<int> workOnCpu(vector<int> rgb);
void RGBtoHSV_CPU (int* rgb, vector<float>& h, vector<float>& s, vector<float>& v,int size);
void HSVtoRGB_CPU(float* h,float *s,float*v,int size,vector<int>&rgb);
int* histo_CPU(float*v,int size);
int* repartition_CPU(int* histo,int size);
vector<float> egalisation_CPU(int* repartition, vector<float> v);


#endif