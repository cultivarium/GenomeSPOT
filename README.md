

Predicts pH, temperature, oxygen, and salinity preferences from bacterial and archaeal genomes using models trained on diverse, phylogenetically balanced sets of isolates.

Reference:
> (reference)

# Quick start
## 1. Install requirements
Requirements for use:
```
biopython
ncbi-genome-download
scikitlearn
hmmlearn
numpy
pandas
```

Requirements only for model training:
```
bacdive
```


## 2. Run prediction

```shell
pip install -r requirements.txt
python src/predict_physicochemistry.py # flags ~#s/genome/CPU
```
Hint: use prodigal for predicting protein FASTA sequences from genomes

```shell
prodigal -i genome.fna -a protein.faa # get proteins, ~#s/genome/CPU
```

Hint: for thousands of genomes, run in parallel using:

```shell
```


## 3. Intrepret output

Each prediction (e.g. optimum temperature) has: 
- **A predicted value**: For oxygen, the predicted value is a probability of oxygen tolerance between [0,1]. For continuous variables, values are in the units C for temperature, %w/v sodium chloride for salinity, and standard units for pH. 
- **A confidence interval**: The lower and upper limits of the confidence interval for that value
- **A warning flag**: raised if the predicted value initially exceeded the sensical range (values observed in published data) and was set to the min or max allowed value. If a warning flag exists, the prediction should be considered suspect unless it's a predicted salinity minimum or optimum at 0, which is common. 

Here is the output for the test genome:

```
```


# Tutorial

`tutorial.ipynb` provides an interactions demonstration of modules in this repo. Briefly:

- The user provides a genome sequence and protein sequences in FASTA format
- Features of sequences in the genome are calculated by `Genome` using the classes `Protein` and `DNA`
- A pretrained model for each condition (optimum temperature, minimum temperature, etc.) estimates the condition from the genome features
- Model training is discussed in a separate section (see: Model training and evaluation), but a couple functions exist to demonstrate the functions used to balance the data phylogenetically

# Key considerations

- The user is **strongly encouraged** (required if I could!) to understand inaccuracies of the model by reading the publication
- The warning is most likely to occur when the organism is very different than organisms in the training dataset and/or was predicted to have few or no extracellular proteins, which are used in predicting salinity and pH. Please also note that extremophiles have atypical amino acid distributions that make predictions slightly less accurate. For example, oxygen tolerance predictions are slightly less accurate at higher temperature.
- A flag enables saving the intermediate output, which can be use to understand errors or to repeat model training.


# Model training and evaluation

Users may be interested in replicating this work using the provided modules and scientific notebooks.

## Download data for training
```shell
# Download BacDive data
vi .bacdive_credentials # username on line 1, password on line 2
python3 src/query_bacdive.py -c .bacdive_credentials -max 171000 -o data/traits/bacdive_data.json

# Download genomes
ncbi-genome-download -s genbank -F 'fasta,protein-fasta' -o data/genomes -A genbank_accessions.txt 'bacteria,archaea'

# Download GTDB taxonomy
wget https://data.gtdb.ecogenomic.org/releases/latest/ar53_taxonomy.tsv.gz
wget https://data.gtdb.ecogenomic.org/releases/latest/bac120_taxonomy.tsv.gz
# optional for analyses:
wget https://data.gtdb.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz
wget https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz
mv *.tsv.gz data/references/.
gunzip data/references/*tsg.gz
```
## Prepare data

## Train models and select best model with cross-validation


## Produce final model on all data