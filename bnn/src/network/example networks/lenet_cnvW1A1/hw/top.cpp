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
 * @file top.cpp
 *
 * HLS Description of the CNV BNN with axi-lite based parameter loading (DoMemInit) 
 * and  dataflow architecture of the image inference (DoCompute).
 * The network uses 1 bit weights and 1 bit activation.
 *
 *****************************************************************************/
#include "config.h"
#include "bnn-library.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;


static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;


unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) 
{
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) 
{
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
    	// do nothing
      break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW
 	stream<ap_uint<64> > instream("DoCompute.instream");
	stream<ap_uint<8*L0_IFM_CH> > instream_bitw("DoCompute.instream_bitw");
#pragma HLS STREAM variable=instream_bitw depth=128
	
	stream<ap_uint<L0_OFM_CH> > convstream1("DoCompute.convstream1");
	stream<ap_uint<L0_OFM_CH> > poolstream1("DoCompute.poolstream1");
#pragma HLS STREAM variable=poolstream1 depth=128

	stream<ap_uint<L1_OFM_CH> > convstream2("DoCompute.convstream2");
	stream<ap_uint<L1_OFM_CH> > poolstream2("DoCompute.poolstream2");
#pragma HLS STREAM variable=poolstream2 depth=128

	stream<ap_uint<64> > fcstream1("DoCompute.fcstream1");
#pragma HLS STREAM variable=fcstream1 depth=128

	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");

	const unsigned int inBits = IMG_DIM*IMG_DIM*IMG_CH*8;
	const unsigned int outBits = L3_MH*16;

	Mem2Stream_Batch<64, inBits/8> (in, instream, numReps);
	StreamingDataWidthConverter_Batch<64, 8, (inBits) / 64> (instream, instream_bitw, numReps);

	// convolutional layers
	ConvLayerValid_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>(instream_bitw, convstream1, weights0, threshs0, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L0_OFM_DIM, 2, L0_OFM_CH>(convstream1, poolstream1, numReps);

	ConvLayerSame_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Recast<XnorMul>>(poolstream1, convstream2, weights1, threshs1, numReps, ap_resource_lut());
	StreamingMaxPoolOddValid_Batch<L1_OFM_DIM, 2, L1_OFM_CH>(convstream2, poolstream2, numReps);

	// fully connected layers
	WidthAdjustedOutputStream<L3_PE, 64, L3_MH / L3_PE>  wa_out(memOutStrm, numReps);
	StreamingFCLayer_Batch<L2_MW, L2_MH, L2_SIMD, L2_PE, Recast<XnorMul>>
	(poolstream2, fcstream1,  weights2, threshs2, numReps, ap_resource_lut());
	StreamingFCLayer_Batch<L3_MW, L3_MH, L3_SIMD, L3_PE, Recast<XnorMul>, Slice<ap_uint<16> >>
	(fcstream1, static_cast<hls::stream<ap_uint<L3_PE>>&>(wa_out), weights3, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

  Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);

}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
