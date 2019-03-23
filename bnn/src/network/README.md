# bnn-pynq

Network topologies for the PYNQ, ZC706 and Ultra96 release, based on the network descriptions reported in the FINN paper as LFC and CNV. Now, different precision are available within each topology (W1A1 and W1A2 for both and additional W2A2 for CNV). Two scripts are located in the root folder:
 
 - "make-hw.sh" launches HLS synthesis and the overlay generation for a given configuration.
 
 - "make-sw.sh" generates shared objects that uses the accelerator for the PYNQ, ZC706 or Ultra96 for a given configuration. It supports the HW accelerator host code or a SW implementation and automatically detects the Board.

This repo also contains one folder per configuration which is structured like this:

 - "<network config>/hw" contains the HLS config header file and top-level .cpp that instantiates all the layers and wraps them into an IP
 - "<network config>/sw" contains the configuration-specific software code that loads the dataset, initializes memories and launches the accelerator.

 To Cross Compile for ARM copy the files `/usr/include/libxlnx_cma.h` and `/usr/lib/libcma.so` from your PYNQ linux to the same directories on your PC. Add the path to `export VIVADOHLS_INCLUDE_PATH=/opt/Xilinx/Vivado/2018.3/include` to `.bashrc`. Source `.bashrc` and Vivado's settings64.sh. Finally run the following to cross compile for ZC706

 `CC=true BOARD=ZC706 ./make-sw.sh lfcW1A1 python_hw`
 