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
 ******************************************************************************/
 
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file stream-tools.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to adapt stream size, 
 *  remove unnecessary streams (padding) and casting
 *
 ******************************************************************************/

#ifndef STREAMTOOLS_H
#define STREAMTOOLS_H

// lookup table defined for alternate padding in StreamPad up to 256 channels
static ap_uint<256> lookuptable("1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010",2);

// function to concatanate 3 three streams, (You can modify it to support any no. of streams. Width of out stream should e NumChannels*streamstoconcat)
template<unsigned int NumChannels, unsigned int Dim, unsigned int Precision = 1>
void ConcatStream(hls::stream<ap_uint<NumChannels * Precision>> & in1 , hls::stream<ap_uint<NumChannels * Precision>> & in2, hls::stream<ap_uint<NumChannels * Precision>> & in3, hls::stream<ap_uint<NumChannels*Precision*3>> &out )
{
	const unsigned int streamstoconcat = 3;
	ap_uint<NumChannels * Precision * streamstoconcat> dataout = 0;
	for (unsigned int i = 0 ; i < Dim*Dim; i++)
	{
#pragma HLS PIPELINE II=1
		dataout(NumChannels*Precision*1-1, NumChannels*Precision*1-NumChannels*Precision) = in1.read();
		dataout(NumChannels*Precision*2-1, NumChannels*Precision*2-NumChannels*Precision) = in2.read();
		dataout(NumChannels*Precision*3-1, NumChannels*Precision*3-NumChannels*Precision) = in3.read();
		out.write(dataout);
	}
}

// concatenating on batch of input
template<unsigned int NumChannels, unsigned int Dim, unsigned int Precision = 1>
void ConcatStream_Batch(hls::stream<ap_uint<NumChannels * Precision>> &in1, hls::stream<ap_uint<NumChannels * Precision>> &in2, hls::stream<ap_uint<NumChannels * Precision>> & in3, hls::stream<ap_uint<NumChannels*Precision*3>> &out, unsigned int numReps)
{
	for(unsigned int i = 0; i < numReps; i++)
	{
		ConcatStream<NumChannels, Dim, Precision>(in1, in2, in3, out);
	}
}

// function to clone stream into any no. (You can add your streams)
template<unsigned int NumChannels, unsigned int Dim, unsigned int Precision = 1>
void CloneStream(hls::stream<ap_uint<NumChannels * Precision>> &in, hls::stream<ap_uint<NumChannels * Precision>> &out1, hls::stream<ap_uint<NumChannels * Precision>> &out2/*, hls::stream<ap_uint<NumChannels * Precision>> &out3*/)
{
	ap_uint<NumChannels * Precision> data;
	for(unsigned int i = 0; i < Dim*Dim; i++)
	{
#pragma HLS PIPELINE II=1
		data = in.read();
		out1.write(data);
		out2.write(data);
		//out3.write(data);
	}
}

// function to operate on a batch on input
template<unsigned int NumChannels, unsigned int Dim, unsigned int Precision = 1>
void CloneStream_Batch(hls::stream<ap_uint<NumChannels * Precision>> &in, hls::stream<ap_uint<NumChannels* Precision>> &out1, hls::stream<ap_uint<NumChannels * Precision>> &out2/*, hls::stream<ap_uint<NumChannels * Precision>> &out3*/, unsigned int numReps)
{
	for(unsigned int i = 0; i < numReps; i++)
	{
		CloneStream<NumChannels, Dim, Precision>(in, out1, out2/*, out3*/);
	}
}

// Reshape input stream to output only useful data when padding is VALID:
// Might drop lines and columns at right and bottom
template<
	unsigned int ImgDim,
	unsigned int NumChannels,
	unsigned int Precision = 1>
void ValidResize(hls::stream<ap_uint<NumChannels *Precision> > &in, hls::stream<ap_uint<NumChannels*Precision> > & out)
{
	constexpr unsigned int drop = 1;
	constexpr unsigned int dropAt = ImgDim - drop;
	for(unsigned int i=0; i< dropAt; i++)
	{
		for(unsigned int j=0; j<ImgDim; j++)
		{
			#pragma HLS PIPELINE II=1
			ap_uint<NumChannels*Precision> data = in.read();

			if(j < dropAt)
				out.write(data);
		}
	}
	for(unsigned int i = 0; i<drop; i++)
	{
		for(unsigned int j=0; j<ImgDim; j++)
		{
			#pragma HLS PIPELINE II=1
			in.read();
		}
	}
}

template<
	unsigned int ImgDim,
	unsigned int NumChannels,
	unsigned int Precision = 1>
void ValidResize_Batch(hls::stream<ap_uint<NumChannels*Precision> > &in, hls::stream<ap_uint<NumChannels*Precision> > &out, const unsigned int numReps)
{
	for (unsigned int rep = 0; rep < numReps; rep++)
	{
		ValidResize<ImgDim, NumChannels, Precision>(in, out);
	}
}

template<
        unsigned int NumChannels,
        unsigned int Precision=1
        >
void StreamPad(hls::stream<ap_uint<NumChannels * Precision> > &in,
        hls::stream<ap_uint<NumChannels * Precision> > &out,
        const unsigned int ImgDim,
        const unsigned int PaddedDim)
{

    // Padding
    const unsigned int Padding = PaddedDim - ImgDim;
    // Padding Up and Left
    const unsigned int PaddingUp = Padding / 2;
    const unsigned int PaddingLeft = Padding / 2;
    // Padding Down and Right (might be 1 element more than up and left in case of odd padding)
    const unsigned int PaddingDown = Padding - PaddingUp;
    const unsigned int PaddingRight = Padding - PaddingLeft;

    ap_uint<NumChannels * Precision> outData, inData;
    // Using lookup table
	ap_uint<NumChannels * Precision> val = lookuptable((NumChannels * Precision)-1,0);

	// Using equation
	//ap_uint<NumChannels> val = (1/3)*((2^(NumChannels-1))*(3+(-1)^NumChannels)-2);

    for(unsigned int y = 0; y < PaddedDim; y++){
        for(unsigned int x = 0; x < PaddedDim; x++)
        {
#pragma HLS PIPELINE II=1

            // Padding Rows
            if(y < PaddingUp || y >= (PaddedDim - PaddingDown))
            {
                if (Precision != 1)
                	outData = 0;
                else
                {
					outData = val;
					val = ~val;
                }
            }
            // Padding Cols
            else if(x < PaddingLeft || x >= (PaddedDim - PaddingRight))
            {
            	if (Precision != 1)
					outData = 0;
				else
				{
					outData = val;
					val = ~val;
				}
            }
            // No Padding
            else
            {
                inData = in.read();
                outData = inData;
            }

            out.write(outData);
        }
    }
}

template<unsigned int NumChannels, unsigned int Precision=1>
void StreamPad_Batch(hls::stream<ap_uint<NumChannels * Precision> > &in, hls::stream<ap_uint<NumChannels * Precision> > &out,
		const unsigned int ImgDim, const unsigned int PaddedDim, unsigned int numReps)
{
	for(unsigned int rep = 0; rep < numReps; rep++)
	{
		StreamPad<NumChannels, Precision>(in, out, ImgDim, PaddedDim);
	}
}

// Reshape input stream to output only useful data when padding is same:
// Might add 0s at left, right, upper, lower side of the input
// Pad with 0
template<unsigned int NumChannels, unsigned int Precision =1>
void StreamPadZero(hls::stream<ap_uint<NumChannels * Precision> > &in, hls::stream<ap_uint<NumChannels * Precision> > &out, const unsigned int ImgDim, const unsigned int PaddedDim)
{

    // Padding
    const unsigned int Padding = PaddedDim - ImgDim;
    // Padding Up and Left
    const unsigned int PaddingUp = Padding / 2;
    const unsigned int PaddingLeft = Padding / 2;
    // Padding Down and Right (might be 1 element more than up and left in case of odd padding)
    const unsigned int PaddingDown = Padding - PaddingUp;
    const unsigned int PaddingRight = Padding - PaddingLeft;

    ap_uint<NumChannels* Precision> outData, inData;

    for(unsigned int y = 0; y < PaddedDim; y++)
    {
        for(unsigned int x = 0; x < PaddedDim; x++)
        {
            #pragma HLS PIPELINE II=1

            // Padding Rows
            if(y < PaddingUp || y >= (PaddedDim - PaddingDown))
            {
                outData = 0;
            }
            // Padding Cols
            else if(x < PaddingLeft || x >= (PaddedDim - PaddingRight))
            {
                outData = 0;
            }
            // No Padding
            else
            {
                inData = in.read();
                outData = inData;
            }

            out.write(outData);
        }
    }
}

template<unsigned int NumChannels, unsigned int Precision = 1>
void StreamPadZero_Batch(hls::stream<ap_uint<NumChannels * Precision> > &in, hls::stream<ap_uint<NumChannels * Precision> > &out,
		const unsigned int ImgDim, const unsigned int PaddedDim, unsigned int numReps)
{
	for(unsigned int rep = 0; rep < numReps; rep++)
	{
		StreamPadZero<NumChannels, Precision>(in, out, ImgDim, PaddedDim);
	}
}

// only let the first X elements of a stream to pass through, the remainder
// are consumed from input but not re-emitted from the output
// useful for getting rid of e.g. padding words
template<unsigned int DataWidth,    // stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal       // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(NumTotal >= NumAllowed);
  unsigned int numLeft = NumAllowed;
  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
    if (numLeft > 0) {
      out.write(e);
      numLeft--;
    }
  }
}

template<unsigned int DataWidth,	// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal       // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter_Batch(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamLimiter<DataWidth, NumAllowed, NumTotal>(in, out);
  }
}

template<typename InT, typename OutT>
void StreamingCast(hls::stream<InT> & in, hls::stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
		hls::stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
  if (InWidth > OutWidth) {
    // emit multiple output words per input word read
    CASSERT_DATAFLOW(InWidth % OutWidth == 0);
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = NumInWords * outPerIn * numReps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read new input word if current out count is zero
      if (o == 0) {
        ei = in.read();
	  }
      // pick output word from the rightmost position
      ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
      out.write(eo);
      // shift input to get new output word for next iteration
      ei = ei >> OutWidth;
      // increment written output count
      o++;
      // wraparound indices to recreate the nested loop structure
      if (o == outPerIn) {
        o = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // straight-through copy
    for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
      ap_uint<InWidth> e = in.read();
      out.write(e);
    }
  } else { // InWidth < OutWidth
    // read multiple input words per output word emitted
    CASSERT_DATAFLOW(OutWidth % InWidth == 0);
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords * numReps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read input and shift into output buffer
      ap_uint<InWidth> ei = in.read();
      eo = eo >> InWidth;
      eo(OutWidth - 1, OutWidth - InWidth) = ei;
      // increment read input count
      i++;
      // wraparound logic to recreate nested loop functionality
      if (i == inPerOut) {
        i = 0;
        out.write(eo);
      }
    }
  }
}

template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedInputStream {
  hls::stream<ap_uint<OW>>  m_target;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
    StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<OW> >&() {
    return  m_target;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedInputStream<W, W, N> {

  hls::stream<ap_uint<W>> &m_source;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<W> >&  source, unsigned const  reps) : m_source(source) {}
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_source;
  }
};

template<unsigned IW, unsigned OW, unsigned N>
class WidthAdjustedOutputStream
{
  hls::stream<ap_uint<IW>>  m_buffer;
  hls::stream<ap_uint<OW>> &m_target;
  unsigned const  m_reps;
  
 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<OW> >&  target, unsigned const  reps) : m_target(target), m_reps(reps) {}
  ~WidthAdjustedOutputStream()
  {
    StreamingDataWidthConverter_Batch<IW, OW, N>(m_buffer, m_target, m_reps);
  }

 public:
  operator hls::stream<ap_uint<IW> >&()
  {
    return  m_buffer;
  }
};
template<unsigned W, unsigned N>
class WidthAdjustedOutputStream<W, W, N>
{
  hls::stream<ap_uint<W>> &m_target;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<W> >&  target, unsigned const  reps)
    : m_target(target) {}
  ~WidthAdjustedOutputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_target;
  }
};
#endif
