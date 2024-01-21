Here, we explain how to train our contrastive learning model. 

1. download all the files from kilthub, here we need to use the following files:

   (1) sim_train_scallop.npy: the similarity matrix between samples in the representative set of Scallop, including all the samples from the data augmentation module.

   (2) sim_train_stringtie.npy: the similarity matrix between samples in the representative set of StringTie, including all the samples from the data augmentation module.

   (3) train_features_withaug_scallop.npy: the set representations of all the samples in the representative set of Scallop, including all the samples from the data augmentation module.

   (4) train_features_withaug_stringtie.npy: the set representations of all the samples in the representative set of StringTie, including all the samples from the data augmentation module.

2. Put all the files above and the scripts `autoparadvisor_train.py` and ``
