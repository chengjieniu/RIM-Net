# RIM-Net
#### [Paper](https://arxiv.org/pdf/2201.12763.pdf) |   [Video](https://www.bilibili.com/video/BV1FY411J7s2/) |   [Slides](https://docs.google.com/presentation/d/12IGI8MFPMK-X8nhnyPgRQFtGNthd-h11EZxU32f8ofQ/edit?usp=sharing)

**RIM-Net: Recursive Implicit Fields for Unsupervised Learning of Hierarchical Shape Structures**<br>
[Chengjie Niu](https://chengjieniu.github.io/), 
[Manyi Li](https://manyili12345.github.io/), 
[Kevin (Kai) Xu](https://kevinkaixu.net/), 
[Hao (Richard) Zhang](https://www.cs.sfu.ca/~haoz/).

## Results
<img src="images/teaser.gif"  width="800" />
<br>
RIM-Net recursively decomposes an input 3D shape into two parts, resulting in a binary
tree hierarchy. Each level of the tree corresponds to an assembly of shape parts, represented as implicit functions, to
reconstruct the input shape.

## Getting Started

### Environment
Python3.6.0, TensorFlow 1.9.0, CUDA 11.2.

### Dataset
We use the ready-to-use dataset provided by BAE-Net, the link is:<br>
https://drive.google.com/file/d/1NvbGIC-XqZGs9pz6wgFwwEPALR-iR8E0/view

### Pre-trained models
The link of pre-trained models is:<br>
https://drive.google.com/drive/folders/1OJSuks0fQ-plFrPH2EuDX58aMAt_pQz8?usp=sharing

### Usage:
1. Change the correct path for 'dataset' and 'checkpoint' in RIM_main.py

2. Set 'train' to 'True' in RIM_main.py for training.

3. Set 'train' to 'False', and 'recon' to 'True' in RIM_mian.py for testing.

If you have more questions, please contact nchengjie@gmail.com.

