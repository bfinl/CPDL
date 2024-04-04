# CPDL
This is a repository of the scripts used for the study: “Continuous Tracking using Deep Learning-based Decoding for Non-invasive Brain-Computer Interface”. 

D. Forenzo, H. Zhu, J. Shanahan, J. Lim, and B. He, “Continuous Tracking using Deep Learning-based Decoding for Non-invasive Brain-Computer Interface.” bioRxiv, p. 2023.10.12.562084, Oct. 17, 2023. doi: 10.1101/2023.10.12.562084.

The files include the EEGNet architecture used in this study, the PointNet architecture, the paradigm used to train the DL models, and a list of electrode locations for the Neuroscan Quik-Cap 64 electrode paradigm that can be used with the PointNet architecture (10-20.txt).

The DL architectures used here are built using the Pytorch framework and are constructed using functions and layers provided by external packages not written by the authors of this manuscript. The EEGNet_like.py model is a version of the EEGNetv4 model from the braindecode package [1], based on the original EEGNet architecture [2], and lightly modified by Hao Zhu. The pointNet2.py model was constructed by Hao Zhu using functions and layers provided by the Pointnet_Pointnet2_pytorch Github repository [3], based on the original PointNet++ architecture [4]. The DL_training.py paradigm was written by Hao Zhu. Additional documentation of files was done by Dylan Forenzo.

[1] braindecode package: 
https://braindecode.org/stable/generated/braindecode.models.EEGNetv4.html 

Reference: R. T. Schirrmeister et al., “Deep learning with convolutional neural networks for EEG decoding and visualization,” Human Brain Mapping, vol. 38, no. 11, pp. 5391–5420, 2017, doi: 10.1002/hbm.23730.

[2] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,” J. Neural Eng., vol. 15, no. 5, p. 056013, Jul. 2018, doi: 10.1088/1741-2552/aace8c.

[3] PointNet_Pointnet2_pytorch repository: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

[4] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.” arXiv, Jun. 07, 2017. Accessed: Apr. 15, 2023. [Online]. Available: http://arxiv.org/abs/1706.02413


