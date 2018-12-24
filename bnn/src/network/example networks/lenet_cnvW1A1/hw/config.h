/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    26  OFM_CH =    32
 *     SIMD  =     1    PE   =    32
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 389376   Ext Latency  =  6084
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 32
#define L0_OFM_DIM 26
#define L0_SIMD 1
#define L0_PE 32
#define L0_WMEM 9
#define L0_TMEM 1
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    13  IFM_CH =    32
 *      OFM  =    13  OFM_CH =    64
 *     SIMD  =    32    PE   =    64
 *     WMEM  =    25   TMEM  =     1
 *     #Ops  = 17305600   Ext Latency  =  4225
**/

#define L1_K 5
#define L1_IFM_CH 32
#define L1_IFM_DIM 13
#define L1_OFM_CH 64
#define L1_OFM_DIM 13
#define L1_SIMD 32
#define L1_PE 64
#define L1_WMEM 25
#define L1_TMEM 1
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Fully-Connected Layer L2:
 *     MatW =  2304 MatH =  1024
 *     SIMD =    64  PE  =    64
 *     WMEM =   576 TMEM =    16
 *     #Ops  = 4718592   Ext Latency  =   576
**/

#define L2_SIMD 64
#define L2_PE 64
#define L2_WMEM 576
#define L2_TMEM 16
#define L2_MW 2304
#define L2_MH 1024
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =  1024 MatH =    64
 *     SIMD =     1  PE  =    64
 *     WMEM =  1024 TMEM =     1
 *     #Ops  = 131072   Ext Latency  =  1024
**/

#define L3_SIMD 1
#define L3_PE 64
#define L3_WMEM 1024
#define L3_TMEM 1
#define L3_MW 1024
#define L3_MH 64
#define L3_WPI 1
#define L3_API 16
#define L3_WPF 0
#define L3_APF 0


#define LL_MH 64
#define IMG_DIM 28
#define IMG_CH 1
#define no_cl 10

#endif //__LAYER_CONFIG_H_

