# autoparadvisor

This repo contains the trained parameter advisor models for Scallop and StringTie. For details on these models, please read our manuscript "Adaptive, sample-specific parameter selection for more accurate transcript assembly". 

Here we provide an example of generating a parameter advisor set for a new RNA-seq sample via our model. 

## Download fastq files

Download fastq files of the sample (e.g. SRA accession number:SRR1023790) from https://sra-explorer.info. 

## Generate advisor set

1. Download Mash from https://mash.readthedocs.io/en/latest/.
2. Install the python dependencies: `pip install -r requirements.txt`, we use python 3.10.12.
3. download the necessary files from kilthub, it includes

   (1)
4. Put all the files from kilthub, from the folder `./files/`, and the scripts `autoparadvisor_contrastive.py`, `MinHash.capnp`, and `advisorset_generator.py` into the same folder, run the command: `python advisorset_generator.py --name SRR1023790 --fastqs SRR1023790_1.fastq.gz SRR1023790_2.fastq.gz --assembler scallop --top 5` (or `python advisorset_generator.py --name SRR1023790 --fastqs SRR1023790_1.fastq.gz SRR1023790_2.fastq.gz --assembler stringtie --top 5`). Here the value of `--top` is the size of the advisor set. top>=5 is recommended. 

5. Our script will output:

   (1) "SRR1023790.msh": Mash sketch of the sample.
   
   (2) a folder `./SRR1023790_scallop_advisorset/` (or `./SRR1023790_stringtie_advisorset/`) and all the parameter candidates are stored there.   
