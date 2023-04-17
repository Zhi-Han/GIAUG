# Architecture Augmentation for Performance Predictor via Graph Isomorphism

This repository is for the paper "Architecture Augmentation for Performance Predictor via Graph Isomorphism" which is accepted by IEEE Transactions on Cybernetics.

## Prerequisites
The codes have been tested on Python 3.6.

- nasbench (see  https://github.com/google-research/nasbench)
- nas_201_api (see https://github.com/D-X-Y/NAS-Bench-201)
- tensorflow (==1.15.0)
- scikit-learn
- matplotlib
- scipy


## Reproducing Results

### NAS-Bench-101
1. Download the [NAS-Bench-101 dataset](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord) and put in the `data` directory in the root folder of this project.
2. Run `python nasbench101_pp.py` to get the ktau of the various performance predictor using training data with different proportions.

### NAS-Bench-201
1. Download the [NAS-Bench-201 dataset](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view?usp=sharing) and put in the `data` directory in the root folder of this project.
2. Run `python nasbench201_pp.py` to get the ktau of the various performance predictor using training data with different proportions.



