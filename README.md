# Genomic-SPOT: prediction of optimum growth conditions from a genome

Predicts pH, temperature (C), oxygen (probability tolerant), and salinity (% w/v NaCl) preferences from bacterial and archaeal genomes using models trained on diverse, phylogenetically balanced sets of isolates.

Reference:
> (reference)

# Quick start
## 1. Install package

Install package:
```
pip install -r requirements.txt
```

Requirements:
- hmmlearn==0.3.0
- scikit-learn==1.2.2
- biopython>=1.81
- numpy>=1.23.5
- pandas>=1.5.3
- bacdive>=0.2 (only used in model training)

Recommended:
- ncbi-genome-download ([github.com/kblin/ncbi-genome-download](https://github.com/kblin/ncbi-genome-download)) for downloading genomes from GenBank
- prodigal ([github.com/hyattpd/Prodigal](https://github.com/hyattpd/Prodigal)) for predicting protein sequences from genomes

## 2. Run prediction

Runtime: ~5-10 seconds per genome

```shell
python -m genomic_spot.genomic_spot --models models \
    --contigs tests/test_data/GCA_000172155.1_ASM17215v1_genomic.fna.gz \
    --proteins tests/test_data/GCA_000172155.1_ASM17215v1_protein.faa.gz \
    --output GCA_000172155.1
```
Hint: if you only have a genome and need a protein FASTA, use prodigal. 

```shell
gunzip genome.fna.gz # requires unzip
prodigal -i genome.fna -a protein.faa # get proteins
gzip genome.fna
```

Hint: for thousands of genomes, run in parallel using:

```shell
```


## 3. Interpret output

Each prediction (e.g. optimum temperature) has: 
- **A predicted value and units**: For oxygen, the predicted value is a probability of oxygen tolerance between [0,1]. We recommend basing decisions on highly confidence predictions only (p<0.25 means obligate anaerobe, p>0.75 means aerobe, between means uncertain). For continuous variables, values are in the units C for temperature, %w/v sodium chloride for salinity, and standard units for pH. 
- **An estimated error**: For continuous variables only. The error is the root mean squared error (RSME) for cross-validation predictions in the training dataset that were within +/-0.5 pH units, +/-1 % NaCl, or +/-5 C of the predicted value. For oxygen, the probability should be used to assess confidence.
- **A novelty detection**: [Novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection) is like outlier detection, except it is based on a training dataset. For each condition, a genome is novel if its features are more unusual that 98% of the training data. On GTDB genomes, we observed ~4% of genomes have unusual features for oxygen and temperature prediction, and ~10-15% of genomes have unusual features for salinity and pH prediction.
- **A warning flag**: raised if the predicted value initially exceeded the sensical range (values observed in published data) and was set to the min or max allowed value. If a warning flag exists, the prediction should be considered suspect unless it's a predicted salinity minimum or optimum at 0, which is common. 

Here is the output for the test genome:

```
                         value     error is_novel       warning           units
temperature_optimum  29.968543  5.835226    False          None               C
temperature_max      38.087803  5.757588    False          None               C
temperature_min      13.166659  7.179667    False          None               C
ph_optimum            7.527958  0.880036    False          None              pH
ph_max                9.509584  1.101179    False          None              pH
ph_min                5.642458  0.994403    False          None              pH
salinity_optimum      0.807071  1.567823    False          None      % w/v NaCl
salinity_max          4.886699  2.768601    False          None      % w/v NaCl
salinity_min                 0  1.393914    False  min_exceeded      % w/v NaCl
oxygen                0.954284      None    False          None  prob. tolerant
```


# Tutorial

`tutorial.ipynb` provides an interactive demonstration of modules in this repo. Briefly:

- The user provides a genome sequence and protein sequences in FASTA format. Protein prediction is not performed by this package because it would multiply the runtime.
- Features of sequences in the genome are calculated by `Genome` using the classes `Protein` and `DNA`
- A pretrained model for each condition (optimum temperature, minimum temperature, etc.) estimates the condition from the genome features
- Model training is discussed in a separate section (see: Model training and evaluation), but a couple functions exist in the tutorial to demonstrate the functions used to balance the data phylogenetically

# Key considerations

- The user is **strongly encouraged** (required if I could!) to understand inaccuracies of the model by reading the publication
- The warning is most likely to occur when the organism is very different than organisms in the training dataset and/or was predicted to have few or no extracellular proteins, which are used in predicting salinity and pH. Please also note that extremophiles have atypical amino acid distributions that make predictions slightly less accurate. For example, oxygen tolerance predictions are slightly less accurate at higher temperature.
- A flag enables saving the intermediate output, which can be use to understand errors or to repeat model training.


# Model training and evaluation

Users may be interested in replicating this work using the provided modules and scientific notebooks.

## Download data for training

Data is downloaded from two resources:

- BacDive API (**instructions for credentials here**: https://api.bacdive.dsmz.de/)
- Genome Taxonomy Database (info: https://gtdb.ecogenomic.org/)

NOTE: this takes a long time.

```shell
# Download BacDive data
vi .bacdive_credentials # username on line 1, password on line 2
MAX_BACDIVE_ID=171000 # UPDATE THIS OVER TIME!!!
python3 src/model_training/download_training_data.py -u $BACDIVE_USERNAME -p $BACDIVE_PASSWORD \
    --max $MAX_BACDIVE_ID \
    -s raw_bacdive_data.json \
    -o traits.json

# Download genomes
ncbi-genome-download -s genbank -F 'fasta,protein-fasta' -o data/genomes -A genbank_accessions.txt 'bacteria,archaea'

# Download GTDB metadata helpful for analysis
wget https://data.gtdb.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz
wget https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz
mv *.tsv.gz data/references/.
gunzip data/references/*tsg.gz
```
## Prepare data

## Train models and select best model with cross-validation

## Produce final model on all data

```shell
python3 -m genomic_spot.model_training.train_models --training_data_filename data/training_data/training_data_20231203.tsv --path_to_models models
```