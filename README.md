# autoparadvisor

This repo contains the trained parameter advisor models for Scallop and StringTie. For details on these models, please read our manuscript "Adaptive, sample-specific parameter selection for more accurate transcript assembly". 

Here we provide an example of generating a parameter advisor set for a new RNA-seq sample via our model. 

## Download fastq files

Download fastq files of the sample (SRA accession number:SRR1023790) from https://sra-explorer.info. 

## Generate advisor set

1. Download Mash from https://mash.readthedocs.io/en/latest/.
2. Install the python dependencies: `pip install -r requirements.txt`, we use python 3.10.12.
3. download the necessary files from kilthub, it includes

   (1)
4. Put all the files from kilthub and the scripts `autoscallop_contrastive.py` and `advisorset_generator.py` into the same folder, run the command: `python advisorset_generator.py --name SRR1023790 --fastqs SRR1023790_1.fastq.gz SRR1023790_2.fastq.gz --assembler scallop` (or `python advisorset_generator.py --name SRR1023790 --fastqs SRR1023790_1.fastq.gz SRR1023790_2.fastq.gz --assembler stringtie`)
