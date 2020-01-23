/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    28  OFM_CH =     6
 *     SIMD  =     1    PE   =     6
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 84672   Ext Latency  =  7056
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 6
#define L0_OFM_DIM 28
#define L0_SIMD 1
#define L0_PE 6
#define L0_WMEM 9
#define L0_TMEM 1
#define L0_WPI 2
#define L0_API 2
#define L0_WPF 6
#define L0_APF 6

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =     6
 *      OFM  =    14  OFM_CH =    16
 *     SIMD  =     3    PE   =     8
 *     WMEM  =    36   TMEM  =     2
 *     #Ops  = 338688   Ext Latency  =  7056
**/

#define L1_K 3
#define L1_IFM_CH 6
#define L1_IFM_DIM 14
#define L1_OFM_CH 16
#define L1_OFM_DIM 14
#define L1_SIMD 3
#define L1_PE 8
#define L1_WMEM 36
#define L1_TMEM 2
#define L1_WPI 2
#define L1_API 2
#define L1_WPF 6
#define L1_APF 6

/**
 * Fully-Connected Layer L2:
 *     MatW =   784 MatH =   128
 *     SIMD =     4  PE  =     4
 *     WMEM =  6272 TMEM =    32
 *     #Ops  = 200704   Ext Latency  =  6272
**/

#define L2_SIMD 4
#define L2_PE 4
#define L2_WMEM 6272
#define L2_TMEM 32
#define L2_MW 784
#define L2_MH 128
#define L2_WPI 2
#define L2_API 2
#define L2_WPF 6
#define L2_APF 6

/**
 * Fully-Connected Layer L3:
 *     MatW =   128 MatH =    64
 *     SIMD =     2  PE  =     1
 *     WMEM =  4096 TMEM =    64
 *     #Ops  = 16384   Ext Latency  =  4096
**/

#define L3_SIMD 2
#define L3_PE 1
#define L3_WMEM 4096
#define L3_TMEM 64
#define L3_MW 128
#define L3_MH 64
#define L3_WPI 2
#define L3_API 2
#define L3_WPF 6
#define L3_APF 6

/**
 * Fully-Connected Layer L4:
 *     MatW =    64 MatH =    64
 *     SIMD =     1  PE  =     4
 *     WMEM =  1024 TMEM =    16
 *     #Ops  =  8192   Ext Latency  =  1024
**/

#define L4_SIMD 1
#define L4_PE 4
#define L4_WMEM 1024
#define L4_TMEM 16
#define L4_MW 64
#define L4_MH 64
#define L4_WPI 2
#define L4_API 16
#define L4_WPF 6
#define L4_APF 0


#define LL_MH 64
#define IMG_DIM 28
#define IMG_CH 1
#define no_cl 10

#endif //__LAYER_CONFIG_H_

