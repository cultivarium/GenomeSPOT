# GenomeSPOT: <u>G</u>enome-based <u>S</u>alinity, <u>p</u>H, <u>O</u>xygen Tolerance, and <u>T</u>emperature for Bacteria and Archaea

Predicts oxygen tolerance (obligate/facultative aerobe vs. obligate anaerobe) and optimum, min, and max temperature (C), salinity (% w/v NaCl), and pH for bacterial and archaeal genomes using statistical models trained on data for phylogenetically diverse microbial isolates.

Reference:
> (reference)

# Quick start
## 1. Install package

Clone the repo, create a virtual environment, then install packages:
```
pip install -r requirements.txt
```

Requirements:
- hmmlearn==0.3.0
- scikit-learn==1.2.2 (The correct scikit-learn version is **ESSENTIAL** to use models)
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
python -m genome_spot.genome_spot --models models \
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

## 3. Interpret output

Each prediction (e.g. optimum temperature) has: 
- **A predicted value and units**: For oxygen, the predicted value is a classification of tolerance as "tolerant" or "not tolerant". For continuous variables, values are in the units C for temperature, %w/v sodium chloride for salinity, and standard units for pH. 
- **An estimated error**: For oxygen, the error is provided as the probability of the classification from 0.5 to 1 (p<0.5 results in the other classification). We recommend basing decisions on highly confidence predictions (p>0.75). For continuous variables only. The error is the root mean squared error (RSME) for cross-validation predictions in the training dataset that were within +/-0.5 pH units, +/-1 % NaCl, or +/-5 C of the predicted value.
- **A novelty detection**: [Novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection) is like outlier detection, except it is based on a training dataset. For each condition, a genome is novel if its features are more unusual that 98% of the training data. On GTDB genomes, we observed ~4% of genomes have unusual features for oxygen and temperature prediction, and ~10-15% of genomes have unusual features for salinity and pH prediction.
- **A warning flag**: raised if the predicted value initially exceeded the sensical range (values observed in published data) and was set to the min or max allowed value. If a warning flag exists, the prediction should be considered suspect unless it's a predicted salinity minimum or optimum at 0, which is common. 

Here is the output for the test genome:
```
                         value     error is_novel       warning        units
temperature_optimum  28.810433   5.68601    False          None            C
temperature_max      37.384123  5.544945    False          None            C
temperature_min      11.745063  6.781135    False          None            C
ph_optimum            7.512092  0.900025    False          None           pH
ph_max                9.547053   1.02375    False          None           pH
ph_min                5.610827    0.9734    False          None           pH
salinity_optimum      0.968301  1.470593    False          None   % w/v NaCl
salinity_max           5.27174  3.464518    False          None   % w/v NaCl
salinity_min                 0  1.393914    False  min_exceeded   % w/v NaCl
oxygen                tolerant   0.95858    False          None  probability
```



# Tutorial

`tutorial.ipynb` provides an interactive demonstration of modules in this repo. Briefly:

- The user provides a genome sequence and protein sequences in FASTA format. Protein prediction is not performed by this package because it would multiply the runtime.
- Features of sequences in the genome are calculated by `Genome` using the classes `Protein` and `DNA`
- A pretrained model for each condition (optimum temperature, minimum temperature, etc.) estimates the condition from the genome features
- Model training is discussed in a separate section (see: Model training and evaluation), but a couple functions exist in the tutorial to demonstrate the functions used to balance and partition the data phylogenetically

# Key considerations

Predictions can be inaccurate:

- The user is **strongly encouraged** (required if I could!) to understand inaccuracies of the model by reading the publication
- The warning is most likely to occur when the organism is very different than organisms in the training dataset and/or was predicted to have few or no extracellular proteins, which are used in predicting salinity and pH. Please also note that extremophiles have atypical amino acid distributions that make predictions slightly less accurate. For example, oxygen tolerance predictions are slightly less accurate at higher temperature.
- A flag enables saving the intermediate output, the features calculated on a genome, which can be use to understand errors or to train other models.


Our models were built using phylogenetic balancing and partitioning to improve accuracy:

- **Phylogenetic balancing** addresses phylogenetic bias or imbalance by removing genomes from taxa that are more common than they should be. Our models were created and evaluated by removing 50% of genomes, preferentially removing taxa more common in BacDive than in the Genome Taxonomy Database. For example, Pseudomonadota over Verrucomicrobiota and Escherichia over Ktedonobacter.
- **Phylogenetic partitioning** addresses "data leakage" caused by phylogenetic similarity by creating test sets with different clades of organisms than those in the training sets. The test set used to evaluate our model was created by selecting random families adding up to 20% of the genomes in the dataset. To ensure extreme values are included in both the training and test dataset, extreme values are split separately, which means that a family can be present in both the training and test dataset, but the family members will have different growth conditions.

# How to use with multiple genomes

If you have made predictions for many genomes and want results in a single table, a helper script is provided to join predictions for multiple genomes into a single TSV (`all.predictions.tsv`). The first and second rows of the table indicate. respectively, the growth condition being predicted (e.g. `temperature_optimum`) and the type of data in the column (e.g. `error`). All other rows contain data per genome.

```python
python3 -m genome_spot.join_outputs --dir data/predictions --write-to-tsv
```

If you have tens of thousands of genomes and no convenient resources for parallelizing computing tasks, a simple option is to use a shell command `pwait x` to perform x processes at once ([reference](https://stackoverflow.com/questions/38160/parallelize-bash-script-with-maximum-number-of-processes/880864#880864)). The below example runs 10 jobs at once.

```shell
# Define pwait
function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

# Run in parallel
INDIR='data/features'
OUTDIR='data/predictions'
for FEATURES_JSON in `ls $INDIR`; 
do
PREFIX=$(echo $FEATURES_JSON | cut -d. -f1);
echo $FEATURES_JSON $PREFIX;
python -m genome_spot.genome_spot --models models --genome-features $INDIR/$FEATURES_JSON --output $OUTDIR/$PREFIX > temp.txt &;
pwait 10
done
```



# How to repeat model training and evaluation

If you are interested in replicating this work, you can use the provided modules and scientific notebooks. Note that the total workflow may involve several days of runtime.

## 1. Download data for training

Data is downloaded from two resources:

- BacDive API (**instructions for credentials here**: https://api.bacdive.dsmz.de/)
- Genome Taxonomy Database (info: https://gtdb.ecogenomic.org/)

Runtime: under a day.

```shell
# Create directory structure
mkdir data
mkdir data/training_data
mkdir data/genomes
mkdir data/references

# Download BacDive data
vi .bacdive_credentials # username on line 1, password on line 2
MAX_BACDIVE_ID=171000 # UPDATE THIS OVER TIME!!!
python3 -m genome_spot.model_training.download_trait_data -u $BACDIVE_USERNAME -p $BACDIVE_PASSWORD \
    --max $MAX_BACDIVE_ID \
    -b data/training_data/bacdive_data.json \
    -o data/training_data/trait_data.tsv \
    -a genbank_accessions.txt

# Download genomes using
# list created by above function
ncbi-genome-download -s genbank -F 'fasta,protein-fasta' -o data/genomes -A genbank_accessions.txt 'bacteria,archaea'

# Download GTDB metadata to provide taxonomy
# needed for modeling correctly
wget https://data.gtdb.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz
wget https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz
mv *.tsv.gz data/references/.
gunzip data/references/*tsg.gz
```

## 2. Generate training dataframe

Measure features from genomes and join them with the target variables to be predicted - i.e. trait data from BacDive - to create the training dataset.

Runtime: under a day for ~10-20k genomes.

```shell
python3 -m genome_spot.model_training.make_training_dataset -p 7 \
    --genomes-directory ./data/genomes/ \
    -sfna .fna.gz -sfaa .faa.gz \
    --features-directory ./data/training_data/genome_features/ \
    --downloaded-traits ./data/training_data/bacdive_data.json \
    --tsv-output ./data/training_data/training_data.tsv
```


## 3. Create train, test, and cross-validation sets

The training dataset should be balanced phylogenetically and partitioned into different sets of taxa for testing and training.

The above script performs an automated curation step. Only genomes for which the `use_<condition>` flag is `True` are used further. You may want to perform additional curation to remove suspect values, as we did for curation. No further curation occurs.

The function `make_holdout_sets` performs phylogenetic balancing and partitioning for test and cross-validation sets:
1. Genomes are balanced and partitioned at the family level into **training and test sets** for each condition being predicted. Genome accessions are recorded in files like `<path_to_holdouts>/train_set_<condition>.txt`
2. Genomes in the training set are further divided for **cross-validation**. In each fold of a cross-validation, a different set of genomes are held out of model training and used to score the model. The default script performs 5-fold cross-validation, for each rank in phylum, class, order, family, genus, and species. Genome accessions are stored in `<path_to_holdouts>/<condition>_cv_sets.json`, keyed by rank and in list of tuples of `(training_indices, validation_indices)`.

Runtime: 1-2 minutes.

```shell
# Create train, test, and cross-validation sets
python3 -m genome_spot.model_training.make_holdout_sets --training_data_filename  data/training_data/training_data.tsv --path_to_holdouts data/holdouts/ --overwrite
```

## 4. Train a variety of models and select the best model

In the preprint, we describe testing different sets of features (e.g. only amino acid frequencies) with different types of models (e.g. [least squares, ridge, and lasso linear regressions](https://scikit-learn.org/stable/modules/linear_model.html)). To reproduce that workflow, you can run the script `run_model_selection.py`. The sets of features and estimators are hard-coded into the script `run_model_selection`, so you will be unable to try new sets of features or models without modifying code.

Runtime: ~10 minutes.

```shell
python3 -m genome_spot.model_training.run_model_selection --training_data_filename data/training_data/training_data.tsv --path_to_holdouts data/holdouts --outdir data/model_selection/
```

The script saves models and statistics to the output directory. The statistics can be assessed and used to select a model using the notebook `evaluate_models.ipynb`

Model selection should result in a set of instructions specifying the features and estimator to be used for each condition. The features are specified by a list of features, and the estimator is specified by a link to the model saved by the above script. These instructions should be saved in a file in the directory where models will be placed, like the existing `models/instructions.json` file. Importantly, the 'pipeline_filename' specified in instructions must be a path that is interpretable when running the below script `train_models`. 

## 5. Produce the final model using all available data

With a selected model specific by `models/instructions.json`, you can produce the final versions of each model. To be as representative as possible, these models are trained on data from both the training and test sets (both of which are phylogenetically balanced). For example, if the pipeline filename is `'./data/model_selection/temperature_optimum_features1_pipeline17-lasso.joblib'`, then `train_models` must be run from where from `data` is an immediate subdirectory.

```shell
python3 -m genome_spot.model_training.train_models --training_data_filename data/training_data/training_data.tsv --path_to_models models --path_to_holdouts data/holdouts
```

## 6. Perform analyses described in preprint

Scientific notebooks provided in `notebooks` assist with reproducing analyses in the preprint. 

Some analyses involved genomes from the Genome Taxonomy Database and JGI Genomic catalog of Earth’s Microbiomes (GEM) catalogue, which can be obtained from these repositories:

- https://data.gtdb.ecogenomic.org/
- https://portal.nersc.gov/GEM/