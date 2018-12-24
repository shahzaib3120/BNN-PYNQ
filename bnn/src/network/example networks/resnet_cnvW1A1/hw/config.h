/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    28  OFM_CH =    64
 *     SIMD  =     1    PE   =    32
 *     WMEM  =    18   TMEM  =     2
 *     #Ops  = 903168   Ext Latency  = 14112
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 64
#define L0_OFM_DIM 28
#define L0_SIMD 1
#define L0_PE 32
#define L0_WMEM 18
#define L0_TMEM 2
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =    64
 *      OFM  =    14  OFM_CH =    64
 *     SIMD  =    32    PE   =    32
 *     WMEM  =    36   TMEM  =     2
 *     #Ops  = 14450688   Ext Latency  =  7056
**/

#define L1_K 3
#define L1_IFM_CH 64
#define L1_IFM_DIM 14
#define L1_OFM_CH 64
#define L1_OFM_DIM 14
#define L1_SIMD 32
#define L1_PE 32
#define L1_WMEM 36
#define L1_TMEM 2
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =    14  IFM_CH =    64
 *      OFM  =    14  OFM_CH =    64
 *     SIMD  =    64    PE   =    64
 *     WMEM  =     9   TMEM  =     1
 *     #Ops  = 14450688   Ext Latency  =  1764
**/

#define L2_K 3
#define L2_IFM_CH 64
#define L2_IFM_DIM 14
#define L2_OFM_CH 64
#define L2_OFM_DIM 14
#define L2_SIMD 64
#define L2_PE 64
#define L2_WMEM 9
#define L2_TMEM 1
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =  3136 MatH =   512
 *     SIMD =    32  PE  =    16
 *     WMEM =  3136 TMEM =    32
 *     #Ops  = 3211264   Ext Latency  =  3136
**/

#define L3_SIMD 32
#define L3_PE 16
#define L3_WMEM 3136
#define L3_TMEM 32
#define L3_MW 3136
#define L3_MH 512
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Fully-Connected Layer L4:
 *     MatW =   512 MatH =    64
 *     SIMD =     1  PE  =     4
 *     WMEM =  8192 TMEM =    16
 *     #Ops  = 65536   Ext Latency  =  8192
**/

#define L4_SIMD 1
#define L4_PE 4
#define L4_WMEM 8192
#define L4_TMEM 16
#define L4_MW 512
#define L4_MH 64
#define L4_WPI 1
#define L4_API 16
#define L4_WPF 0
#define L4_APF 0


#define LL_MH 64
#define IMG_DIM 28
#define IMG_CH 1
#define no_cl 10

#endif //__LAYER_CONFIG_H_

