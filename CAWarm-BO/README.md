# CAWarm-BO

Here we describe the way to run CAWarm-BO on RNA-seq samples. 

## Installation 

The codes here heavily depend on the codes in https://github.com/xingchenwan/Casmopolitan. Therefore, please follow their documents for installing the required packages. 

In addition, we need to install the following packages and softwares: 

1. PyYAML, `pip install PyYAML`
2. Transcript assemblers such as Scallop (https://github.com/Kingsford-Group/scallop) and StringTie (https://ccb.jhu.edu/software/stringtie/).
3. gffcompare: https://github.com/gpertea/gffcompare
4. gtfcuff: https://github.com/Kingsford-Group/rnaseqtools

## Setting YAML config 

First, we need to download a gene annotation file (.gtf), e.g. https://www.gencodegenes.org/human/release_24.html. This file is for computing AUC values. 

