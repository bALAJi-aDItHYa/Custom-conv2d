# Custom-conv2d
I am trying to implement a custom Conv2d operation where the default multiplication is replaced with a multiplication algo of my own.

The training and testing is being done over the MNIST dataset. The list of src files are:
1) custom_conv2d.py - implementation of custom convolution
2) Net_py.py - Network architecture class defined here
3) MBM.py - Implementation of diff mult algorithm (HDL related!)
4) main.py - training, validation and testing of the architecture over MNIST dataset is initiated here
