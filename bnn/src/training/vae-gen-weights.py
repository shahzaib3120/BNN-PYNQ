#BSD 3-Clause License
#=======
#
#Copyright (c) 2017, Xilinx
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from finnthesizer import *

if __name__ == "__main__":

    bnnRoot = "."
    npzFile = bnnRoot + "/weights/vae_parameters.npz"
    targetDirBin = bnnRoot + "/binparam-cnvW1A1-pynq"
    targetDirHLS = bnnRoot + "/binparam-cnvW1A1-pynq/hw"
    
    #topology of convolutional layers (only for config.h defines)
    ifm       = [28, 13,  5,  0,  2,  6, 14, 0]
    ofm       = [13,  5,  2,  0,  6, 14, 30, 0]   
    ifm_ch    = [ 1, 64, 64,  0, 64, 64, 64, 0]
    ofm_ch    = [64, 64, 64,  0, 64, 64,  1, 0]   
    filterDim = [ 4,  4,  4,  0,  4,  4,  4, 0]

    WeightsPrecisions_fractional =    [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
    ActivationPrecisions_fractional = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
    InputPrecisions_fractional =      [7 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
    WeightsPrecisions_integer =       [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]
    ActivationPrecisions_integer =    [1 , 1 , 1 , 1 , 1 , 1 , 1 , 16]
    InputPrecisions_integer =         [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]

    classes = map(lambda x: str(x), range(10))
    #configuration of PE and SIMD counts
    peCounts = [64, 64, 64, 64, 64, 64, 1, 64]
    simdCounts = [1, 64, 64, 64, 64, 64, 64, 1]
    num_classes = 10
    if not os.path.exists(targetDirBin):
      os.mkdir(targetDirBin)
    if not os.path.exists(targetDirHLS):
      os.mkdir(targetDirHLS)    

    #read weights
    rHW = BNNWeightReader(npzFile, True)

    config = "/**\n"
    config+= " * Finnthesizer Config-File Generation\n";
    config+= " *\n **/\n\n"
    config+= "#ifndef __LAYER_CONFIG_H_\n#define __LAYER_CONFIG_H_\n\n"

    # process convolutional layers
    for layer in range(0, 8):
      peCount = peCounts[layer]
      simdCount = simdCounts[layer]
      WPrecision_fractional = WeightsPrecisions_fractional[layer]
      APrecision_fractional = ActivationPrecisions_fractional[layer]
      IPrecision_fractional = InputPrecisions_fractional[layer]
      WPrecision_integer = WeightsPrecisions_integer[layer]
      APrecision_integer = ActivationPrecisions_integer[layer]
      IPrecision_integer = InputPrecisions_integer[layer]
      print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, layer)
      if layer == 0:
        # use fixed point weights for the first layer
        (w,t) = rHW.readConvBNComplex(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, WPrecision_integer, APrecision_integer, IPrecision_integer, usePopCount=False)
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (layer, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        print "IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional)

        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, IPrecision_integer, WPrecision_fractional, APrecision_fractional, IPrecision_fractional, numThresBits=24, numThresIntBits=16)
        m.addMatrix(w,t,paddedW,paddedH)


        config += (printConvDefines("L%d" % layer, filterDim[layer], ifm_ch[layer], ifm[layer], ofm_ch[layer], ofm[layer], simdCount, peCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 

        #generate HLS weight and threshold header file to initialize memory directly on bitstream generation       
        #m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(layer) + ".h", str(layer))

        #generate binary weight and threshold files to initialize memory during runtime
        #because HLS might not work for very large header files        
        m.createBinFiles(targetDirBin, str(layer))

      elif layer == 1 or layer == 2 or layer == 4 or layer == 5 or layer == 6:
        # regular binarized layer
        (w,t) = rHW.readConvBNComplex(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, WPrecision_integer, APrecision_integer, IPrecision_integer)
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (layer, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        print "IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional)
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, IPrecision_integer, WPrecision_fractional, APrecision_fractional, IPrecision_fractional)
        m.addMatrix(w,t,paddedW,paddedH)

        config += (printConvDefines("L%d" % layer, filterDim[layer], ifm_ch[layer], ifm[layer], ofm_ch[layer], ofm[layer], simdCount, peCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 

        #generate HLS weight and threshold header file to initialize memory directly on bitstream generation        
        #m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(layer) + ".h", str(layer))

        #generate binary weight and threshold files to initialize memory during runtime
        #because HLS might not work for very large header files        
        m.createBinFiles(targetDirBin, str(layer))
 
      else:
	peCount = peCounts[layer]
	simdCount = simdCounts[layer]
	WPrecision_fractional = WeightsPrecisions_fractional[layer]
	APrecision_fractional = ActivationPrecisions_fractional[layer]
	IPrecision_fractional = InputPrecisions_fractional[layer]
	WPrecision_integer = WeightsPrecisions_integer[layer]
	APrecision_integer = ActivationPrecisions_integer[layer]
	IPrecision_integer = InputPrecisions_integer[layer]
	print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, layer)
	(w,t) =  rHW.readFCBNComplex(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, WPrecision_integer, APrecision_integer, IPrecision_integer)
	# compute the padded width and height
	paddedH = padTo(w.shape[0], peCount)
	if (layer == 7):
		paddedH = padTo(w.shape[0], 64)
	paddedW = padTo(w.shape[1], simdCount)
	# compute memory needed for weights and thresholds
	neededWMem = (paddedW * paddedH) / (simdCount * peCount)
	neededTMem = paddedH / peCount
	print "Layer %d: %d x %d" % (layer, paddedH, paddedW)
	print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
	print "IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional)
	m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, IPrecision_integer, WPrecision_fractional, APrecision_fractional, IPrecision_fractional)
	m.addMatrix(w,t,paddedW,paddedH)

	config += (printFCDefines("L%d" % layer, simdCount, peCount, neededWMem, neededTMem, paddedW, paddedH, WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 

	#generate HLS weight and threshold header file to initialize memory directly on bitstream generation
	#if (layer == conv_layers + fc_layers - 1):
	#	m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(layer) + ".h", str(layer), writethreshs = False)
	#else:
	#	m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(layer) + ".h", str(layer))

	#generate binary weight and threshold files to initialize memory during runtime
	#because HLS might not work for very large header files        
	m.createBinFiles(targetDirBin, str(layer))

    config+="\n#define LL_MH %d" %paddedH
    config+="\n#define IMG_DIM %d" %ifm[0]
    config+="\n#define IMG_CH %d" %ifm_ch[0]
    config+="\n#define no_cl %d" %num_classes
    config+="\n\n#endif //__LAYER_CONFIG_H_\n\n"

    configFile = open(targetDirHLS+"/config.h", "w")
    configFile.write(config)
    configFile.close()

    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))

