//for __syncthreads()
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>