/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    28  OFM_CH =    16
 *     SIMD  =     1    PE   =    16
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 225792   Ext Latency  =  7056
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 16
#define L0_OFM_DIM 28
#define L0_SIMD 1
#define L0_PE 16
#define L0_WMEM 9
#define L0_TMEM 1
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =    16    PE   =    32
 *     WMEM  =     1   TMEM  =     1
 *     #Ops  = 200704   Ext Latency  =   196
**/

#define L1_K 1
#define L1_IFM_CH 16
#define L1_IFM_DIM 14
#define L1_OFM_CH 32
#define L1_OFM_DIM 14
#define L1_SIMD 16
#define L1_PE 32
#define L1_WMEM 1
#define L1_TMEM 1
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =    16    PE   =    32
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 1806336   Ext Latency  =  1764
**/

#define L2_K 3
#define L2_IFM_CH 16
#define L2_IFM_DIM 14
#define L2_OFM_CH 32
#define L2_OFM_DIM 14
#define L2_SIMD 16
#define L2_PE 32
#define L2_WMEM 9
#define L2_TMEM 1
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Convolutional Layer L3:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =    16    PE   =    32
 *     WMEM  =    25   TMEM  =     1
 *     #Ops  = 5017600   Ext Latency  =  4900
**/

#define L3_K 5
#define L3_IFM_CH 16
#define L3_IFM_DIM 14
#define L3_OFM_CH 32
#define L3_OFM_DIM 14
#define L3_SIMD 16
#define L3_PE 32
#define L3_WMEM 25
#define L3_TMEM 1
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Fully-Connected Layer L4:
 *     MatW =  4704 MatH =   512
 *     SIMD =    48  PE  =    64
 *     WMEM =   784 TMEM =     8
 *     #Ops  = 4816896   Ext Latency  =   784
**/

#define L4_SIMD 48
#define L4_PE 64
#define L4_WMEM 784
#define L4_TMEM 8
#define L4_MW 4704
#define L4_MH 512
#define L4_WPI 1
#define L4_API 1
#define L4_WPF 0
#define L4_APF 0

/**
 * Fully-Connected Layer L5:
 *     MatW =   512 MatH =    64
 *     SIMD =     1  PE  =    64
 *     WMEM =   512 TMEM =     1
 *     #Ops  = 65536   Ext Latency  =   512
**/

#define L5_SIMD 1
#define L5_PE 64
#define L5_WMEM 512
#define L5_TMEM 1
#define L5_MW 512
#define L5_MH 64
#define L5_WPI 1
#define L5_API 16
#define L5_WPF 0
#define L5_APF 0


#define LL_MH 64
#define IMG_DIM 28
#define IMG_CH 1
#define no_cl 10

#endif //__LAYER_CONFIG_H_

