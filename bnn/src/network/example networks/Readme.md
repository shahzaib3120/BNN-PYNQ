# Running example Networks

Copy the files from the example network you want to run to cnvW1A1, modify clone stream functions accordingly in streamtools.h for resent or inception and change the make-hw.sh script to pass the test image and result and proceed the same way.

For MNIST you can use testimage = 3.bin and result = 3 for inception, resnet and lenet examlpe and for cifar100 testimage = cifar100_single.bin and result = 4

You can also check multiple inference too by setting the single flag to false in main_python.cpp and providing the test image as mnist_15.bin or cifar100_20.bin (labels provided too).

(Note: For MNIST you have to manually repalce the header.num_items to 15 in the parse_mnist_images funciton of xilinx-tiny-cnn/tiny_cnn/io/mnist_parser.h)

For generating .so files on the PYNQ board you need to copy the same files to cnvW1A1 and then run make-sw.sh