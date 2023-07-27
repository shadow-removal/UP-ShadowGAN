# UP-ShadowGAN
Recreating Brightness from Remote Sensing Shadow.

This is the code of the implementation of the UP-ShadowGAN.

# Training
1. Put the training data to corresponding folders (shadow image to ./shadow_URSSR/shadow_train, shadow-free image to ./shadow_URSSR/shadow_free, and test image to ./shadow_URSSR/shadow_test)
2. Python train_UP-ShadowGAN.py

# Testing
1. Python test.py
2. Find the result in corresponding folder (./output/A-shadow generation, B-shadow removal, and Mask)

# Unpaired Remote Sensing Shadow Removal Dataset (URSSR)
Download URSSR from Baidu Cloud: https://pan.baidu.com/s/1yeHS8IHkM15OTafW65DBLg?pwd=1004 key: 1004

# chroma-contrast metric (2C)
Please refer to the 2C folder.

# Acknowledgments
Code is implemented based on https://github.com/xw-hu/Mask-ShadowGAN.

Metric is implemented based on https://ieeexplore.ieee.org/document/7300447 (UCIQE) and https://github.com/imfing/CEIQ (CEIQ).
