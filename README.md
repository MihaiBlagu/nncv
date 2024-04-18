# Final Assignment

This repository is the *messy* implementation of the 5LSM0 final assignemt.

Student: Alexandru-Mihai Blagu
Email: a.blagu@student.tue.nl
Codlab username: mblagu

By default, runnning the `train.py` file runs a training session with the DeepLabV3+ mode (https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py).

Additionally, If a "robustness" training session is to be run, the Alubumenations library has to be installed:

`conda install -c conda-forge albumentations`

Also, the preprocessing function to the cityscapes dataset should be changed from `preprocess_train` to `preprocess_train_robust` in the file train.py at line 48.