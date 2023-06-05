#!/bin/zsh

# Given a list of genomes, tries to gunzip genome, then runs prodigal to
# provide a protein annotation.

FILE_LIST=$1

COUNTER=0
for FILE in `cat ${FILE_LIST}`; do 
GENOME="${FILE/.gz/}"; 
PROTEIN="${FILE/_genomic.fna.gz/_protein.faa}"; 
gunzip $FILE; 
prodigal -q -a $PROTEIN -i $GENOME > prodigal.stdout; # runs quietly
gzip -f $PROTEIN; 
gzip -f $GENOME; 
echo $COUNTER $PROTEIN".gz";
(( COUNTER++ ));
done