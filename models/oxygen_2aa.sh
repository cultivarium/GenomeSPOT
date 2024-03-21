#!/bin/bash

# Predicts aerobe or anaerobe based on different 2 amino acid combinations
# each about 90% accurate but likely flawed in different ways.

echo "O2_CH\tO2_CW\tO2_CE\tC\tH\tW\tE\taas_corrected\tgenome" > oxygen.txt; 
GENOMES_DIR=$1;
for PROTEIN_FASTA in `ls ${GENOMES_DIR}/*faa`; 
do AA_SEQUENCE=$(grep -v '>' $PROTEIN_FASTA | tr -d '*' | tr -d '\n'); 
# count amino acids, removing the n terminal methionine as in preprint
N_AAS=$(echo $AA_SEQUENCE | wc -m | tr -d ' '); 
N_PROTEINS=$(grep -c '>' $PROTEIN_FASTA); 
N_AAS_CORRECTED=$(expr $N_AAS - $N_PROTEINS); 
N_CYS=$(echo "$AA_SEQUENCE" | grep -o 'C' | wc -l | tr -d ' '); 
N_HIS=$(echo "$AA_SEQUENCE" | grep -o 'H' | wc -l | tr -d ' '); 
N_TRY=$(echo "$AA_SEQUENCE" | grep -o 'W' | wc -l | tr -d ' '); 
N_GLU=$(echo "$AA_SEQUENCE" | grep -o 'E' | wc -l | tr -d ' '); 
# equations based on logistic regression
PRED_CH=$(echo "(($N_HIS/$N_AAS_CORRECTED)-0.0043)/($N_CYS/$N_AAS_CORRECTED) > 1.4048" | bc -l); 
PRED_CW=$(echo "(($N_TRY/$N_AAS_CORRECTED)+0.0011)/($N_CYS/$N_AAS_CORRECTED) > 1.1212" | bc -l); 
PRED_CE=$(echo "(($N_GLU/$N_AAS_CORRECTED)-0.1060)/($N_CYS/$N_AAS_CORRECTED) < -3.4803" | bc -l); 
echo "${PRED_CH}\t${PRED_CW}\t${PRED_CE}\t${N_CYS}\t${N_HIS}\t${N_AAS_CORRECTED}\t${PROTEIN_FASTA}" >> oxygen.txt; 
done; 
cat oxygen.txt