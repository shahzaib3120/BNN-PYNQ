/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    32  IFM_CH =     3
 *      OFM  =    30  OFM_CH =    64
 *     SIMD  =     3    PE   =    16
 *     WMEM  =    36   TMEM  =     4
 *     #Ops  = 3110400   Ext Latency  = 32400
 **/

#define L0_K 3
#define L0_IFM_CH 3
#define L0_IFM_DIM 32
#define L0_OFM_CH 64
#define L0_OFM_DIM 30
#define L0_SIMD 3
#define L0_PE 16
#define L0_WMEM 36
#define L0_TMEM 4
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    30  IFM_CH =    64
 *      OFM  =    28  OFM_CH =    64
 *     SIMD  =    32    PE   =    32
 *     WMEM  =    36   TMEM  =     2
 *     #Ops  = 57802752   Ext Latency  = 28224
 **/

#define L1_K 3
#define L1_IFM_CH 64
#define L1_IFM_DIM 30
#define L1_OFM_CH 64
#define L1_OFM_DIM 28
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
 *      OFM  =    12  OFM_CH =   128
 *     SIMD  =    32    PE   =    16
 *     WMEM  =   144   TMEM  =     8
 *     #Ops  = 21233664   Ext Latency  = 20736
 **/

#define L2_K 3
#define L2_IFM_CH 64
#define L2_IFM_DIM 14
#define L2_OFM_CH 128
#define L2_OFM_DIM 12
#define L2_SIMD 32
#define L2_PE 16
#define L2_WMEM 144
#define L2_TMEM 8
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Convolutional Layer L3:
 *      IFM  =    12  IFM_CH =   128
 *      OFM  =    10  OFM_CH =   128
 *     SIMD  =    32    PE   =    16
 *     WMEM  =   288   TMEM  =     8
 *     #Ops  = 29491200   Ext Latency  = 28800
 **/

#define L3_K 3
#define L3_IFM_CH 128
#define L3_IFM_DIM 12
#define L3_OFM_CH 128
#define L3_OFM_DIM 10
#define L3_SIMD 32
#define L3_PE 16
#define L3_WMEM 288
#define L3_TMEM 8
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Fully-Connected Layer L4:
 *     MatW =  3200 MatH =   256
 *     SIMD =     4  PE  =     4
 *     WMEM = 51200 TMEM =    64
 *     #Ops  = 1638400   Ext Latency  = 51200
 **/

#define L4_SIMD 4
#define L4_PE 4
#define L4_WMEM 51200
#define L4_TMEM 64
#define L4_MW 3200
#define L4_MH 256
#define L4_WPI 1
#define L4_API 1
#define L4_WPF 0
#define L4_APF 0

/**
 * Fully-Connected Layer L5:
 *     MatW =   256 MatH =    64
 *     SIMD =     1  PE  =     4
 *     WMEM =  4096 TMEM =    16
 *     #Ops  = 32768   Ext Latency  =  4096
 **/

#define L5_SIMD 1
#define L5_PE 4
#define L5_WMEM 4096
#define L5_TMEM 16
#define L5_MW 256
#define L5_MH 64
#define L5_WPI 1
#define L5_API 16
#define L5_WPF 0
#define L5_APF 0

#define LL_MH 64
#define IMG_DIM 32
#define IMG_CH 3
#define no_cl 10

#endif //__LAYER_CONFIG_H_
