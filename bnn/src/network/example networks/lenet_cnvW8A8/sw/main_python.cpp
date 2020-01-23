/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
 
/******************************************************************************
 *
 *
 * @file main_python.cpp
 *
 * Host code for BNN, overlay cnvW1A1-Pynq, to manage parameter loading and 
 * classification (inference) on single or multiple images.
 * 
 *
 *****************************************************************************/
 
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <algorithm>
#include "config.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void makeNetwork(network<mse, adagrad> & nn) 
{
  nn
#ifdef OFFLOAD
    << chaninterleave_layer<identity>(IMG_CH, IMG_DIM * IMG_DIM, false)
    << offloaded_layer(IMG_CH * IMG_DIM * IMG_DIM, no_cl, &FixedFoldedMVOffload<8, 1, ap_int<32>>, 0xdeadbeef, 0)
#endif
  ;
}

extern "C" void load_parameters(const char* path) 
{
  
  FoldedMVInit("cnvW8A8");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  cout << "Setting network weights and thresholds in accelerator..." << endl;
  FoldedMVLoadLayerMem(path, 0, L0_PE, L0_WMEM, L0_TMEM, 128);
  FoldedMVLoadLayerMem(path, 1, L1_PE, L1_WMEM, L1_TMEM, 128);
  FoldedMVLoadLayerMem(path, 2, L2_PE, L2_WMEM, L2_TMEM, 128);
  FoldedMVLoadLayerMem(path, 3, L3_PE, L3_WMEM, L3_TMEM, 128);
  FoldedMVLoadLayerMem(path, 4, L4_PE, L4_WMEM, L4_TMEM, 0);

}

extern "C" int inference(const char* path, int results[LL_MH], int number_class, float* usecPerImage) {
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> class_result;
  float usecPerImage_int;

  FoldedMVInit("cnvW8A8");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  class_result=testPrebuiltCIFAR10_from_image<8, 32, ap_int<32>>(test_images, number_class, usecPerImage_int);

  if(results) {
    std::copy(class_result.begin(),class_result.end(), results);
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return (std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())));
}

extern "C" int* inference_multiple(const char* path, int number_class, int* image_number, float* usecPerImage, int enable_detail = 0) {
  std::vector<int> detailed_results;
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> all_result;
  float usecPerImage_int;
  int * result;

  FoldedMVInit("cnvW8A8");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  all_result=testPrebuiltCIFAR10_multiple_images<8, 32, ap_int<32>>(test_images, number_class, detailed_results, usecPerImage_int);

  if (image_number) {
    *image_number = all_result.size();
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  if (enable_detail) {
    result = new int [detailed_results.size()];
    std::copy(detailed_results.begin(),detailed_results.end(), result);
  } else {
    result = new int [all_result.size()];
    std::copy(all_result.begin(),all_result.end(), result);
  }	   
  return result;
}

extern "C" void free_results(int* result) {
  delete[] result;
}

extern "C" void deinit() {
  FoldedMVDeinit();
}

extern "C" int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "4 parameters are needed: " << endl;
    cout << "1 - folder for the binarized weights (binparam-***) - full path " << endl;
    cout << "2 - path to image to be classified" << endl;
    cout << "3 - number of classes in the dataset" << endl;
    cout << "4 - expected result" << endl;
    return 1;
  }
  float execution_time = 0;
  int scores[LL_MH];
  load_parameters(argv[1]);
  cout<<"Loading parameters from : "<<argv[1]<<endl;
  bool single = true;
  if(single)
  {
		int class_inference = inference(argv[2], scores, no_cl, &execution_time);
		cout << "Detected class " << class_inference << endl;
		cout << "in " << execution_time << " microseconds" << endl;
		deinit();
		if (class_inference != atol(argv[4]))
		{return 1;}
		else
    {return 0;}
  }
	else
	{
		// for checking multiple inference
		int ex_no = 15;
		int *class_inference = inference_multiple(argv[2], no_cl, &ex_no, &execution_time);
		for(int i = 0 ; i < ex_no; i++)
		{
			cout << "Results = "<< class_inference[i] << endl;
		}
    deinit();
		return 0;
	}
}
