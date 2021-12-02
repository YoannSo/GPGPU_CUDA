#ifndef __WORK_ON_GPU_HPP
#define __WORK_ON_GPU_HPP

#include "utils/common.hpp"
#include "utils/chronoGPU.hpp"
#include <iomanip> 
#include <iostream>
#include <vector>
using namespace std;
__host__ int* workOnGpu(vector<int> rgb);
__global__ void RGBtoHSV_GPU(int* dev_rgb,float* dev_h,float* dev_s,float* dev_v,int rgbSize);
__global__ void computeHistogram_GPU(const float* dev_v, int* dev_repartition, const int v_size);
__global__ void repartition_GPU(int* dev_histo, int* dev_repartition, int size);
__global__ void egalisation_GPU(const int* dev_repartition, float* dev_v,float* vbis, const int v_size);
__global__ void HSVtoRGB_GPU(float* h,float* s,float* v,int* rgb,int HSVsize);
__global__ void computeHistogram_GPU_sharedMemoryVersion(const float* dev_v, int* dev_histo, const int v_size);

#endif