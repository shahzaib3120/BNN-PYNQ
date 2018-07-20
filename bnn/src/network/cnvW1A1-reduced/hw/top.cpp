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

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM> weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM> weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM> weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM> weights3;
static BinaryWeights<L4_SIMD, L4_PE, L4_WMEM> weights4;
static BinaryWeights<L5_SIMD, L5_PE, L5_WMEM> weights5;

static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API>> threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>> threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>> threshs2;
static ThresholdsActivation<L3_TMEM, L3_PE, L3_API, ap_int<16>, ap_uint<L3_API>> threshs3;
static ThresholdsActivation<L4_TMEM, L4_PE, L4_API, ap_int<16>, ap_uint<L4_API>> threshs4;

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo)
{
    if (in % padTo == 0)
    {
        return in;
    }
    else
    {
        return in + padTo - (in % padTo);
    }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val)
{
    switch (targetLayer)
    {
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
        threshs3.m_thresholds[targetMem][targetInd][targetThresh] = val;
        break;
    case 8:
        weights4.m_weights[targetMem][targetInd] = val;
        break;
    case 9:
        threshs4.m_thresholds[targetMem][targetInd][targetThresh] = val;
        break;
    case 10:
        weights5.m_weights[targetMem][targetInd] = val;
        break;
    case 11:
        // do nothing, no thres mem for layer 5 as PassThrough activation is used
        break;
    }
}

void DoCompute(ap_uint<64> *in, ap_uint<64> *out, const unsigned int numReps)
{
#pragma HLS DATAFLOW
    stream<ap_uint<64>> inter0("DoCompute.inter0");
    stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
    stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable = inter0_2 depth = 128
    stream<ap_uint<L0_OFM_CH>> conv1("DoCompute.conv1");
#pragma HLS STREAM variable = conv1 depth = 128
    stream<ap_uint<L1_OFM_CH>> conv2("DoCompute.conv2");
    stream<ap_uint<64>> batch1("DoCompute.batch1");
#pragma HLS STREAM variable = batch1 depth = 128
    stream<ap_uint<L2_OFM_CH>> conv3("DoCompute.conv3");
#pragma HLS STREAM variable = conv3 depth = 128
    stream<ap_uint<L3_OFM_CH>> conv4("DoCompute.conv4");
    stream<ap_uint<128>> batch2("DoCompute.batch2");
#pragma HLS STREAM variable = batch2 depth = 81
    stream<ap_uint<256>> lfc1("DoCompute.lfc1");
#pragma HLS STREAM variable = lfc1 depth = 64
    //     stream<ap_uint<256>> lfc2("DoCompute.lfc2");
    // #pragma HLS STREAM variable = inter10 depth = 3
    stream<ap_uint<64>> memOutStrm("DoCompute.memOutStrm");

    const unsigned int inBits = 32 * 32 * 3 * 8;
    // const unsigned int inBitsPadded = paddedSize(inBits, 64);
    const unsigned int outBits = LL_MH * 16;

    Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);
    StreamingDataWidthConverter_Batch<64, 192, (32 * 32 * 3 * 8) / 64>(inter0, inter0_1, numReps);
    StreamingDataWidthConverter_Batch<192, 24, (32 * 32 * 3 * 8) / 192>(inter0_1, inter0_2, numReps);
    // convolutional layers
    ConvLayerValid_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>(inter0_2, conv1, weights0, threshs0, numReps, ap_resource_lut());
    ConvLayerValid_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Recast<XnorMul>>(conv1, conv2, weights1, threshs1, numReps, ap_resource_lut());

    StreamingMaxPoolEven_Batch<L1_OFM_DIM, 2, L1_OFM_CH>(conv2, batch1, numReps);

    ConvLayerValid_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Recast<XnorMul>>(batch1, conv3, weights2, threshs2, numReps, ap_resource_lut());
    ConvLayerValid_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, Recast<XnorMul>>(conv3, conv4, weights3, threshs3, numReps, ap_resource_lut());

    StreamingMaxPoolEven_Batch<L3_OFM_DIM, 2, L3_OFM_CH>(conv4, batch2, numReps);

    // fully connected layers
    WidthAdjustedOutputStream<16 * L5_PE, 64, L5_MH / L5_PE> wa_out(memOutStrm, numReps);
    StreamingFCLayer_Batch<L4_MW, L4_MH, L4_SIMD, L4_PE, Recast<XnorMul>>(batch2, lfc1, weights4, threshs4, numReps, ap_resource_lut());
    // StreamingFCLayer_Batch<L7_MW, L7_MH, L7_SIMD, L7_PE, Recast<XnorMul>>(inter9, inter10, weights7, threshs7, numReps, ap_resource_lut());
    StreamingFCLayer_Batch<L5_MW, L5_MH, L5_SIMD, L5_PE, Recast<XnorMul>, Slice<ap_uint<16>>>(lfc1, static_cast<hls::stream<ap_uint<16 * L5_PE>> &>(wa_out), weights5, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

    Stream2Mem_Batch<64, outBits / 8>(memOutStrm, out, numReps);
}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
                 unsigned int targetLayer, unsigned int targetMem,
                 unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps)
{
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS INTERFACE s_axilite port = doInit bundle = control
#pragma HLS INTERFACE s_axilite port = targetLayer bundle = control
#pragma HLS INTERFACE s_axilite port = targetMem bundle = control
#pragma HLS INTERFACE s_axilite port = targetInd bundle = control
#pragma HLS INTERFACE s_axilite port = targetThresh bundle = control
#pragma HLS INTERFACE s_axilite port = val bundle = control
#pragma HLS INTERFACE s_axilite port = numReps bundle = control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset = slave port = in bundle = hostmem depth = 512
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE m_axi offset = slave port = out bundle = hostmem depth = 16
#pragma HLS INTERFACE s_axilite port = out bundle = control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable = weights0.m_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs0.m_thresholds complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs0.m_thresholds complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weights1.m_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs1.m_thresholds complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs1.m_thresholds complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weights2.m_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs2.m_thresholds complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs2.m_thresholds complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weights3.m_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs3.m_thresholds complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs3.m_thresholds complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weights4.m_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs4.m_thresholds complete dim = 1
#pragma HLS ARRAY_PARTITION variable = threshs4.m_thresholds complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weights5.m_weights complete dim = 1

    if (doInit)
    {
        DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
    }
    else
    {
        DoCompute(in, out, numReps);
    }
}
