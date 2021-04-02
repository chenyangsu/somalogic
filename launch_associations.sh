#!/bin/bash

#PBS -o logs/
#PBS -e logs/
#PBS -l mem=7G
#PBS -l vmem=7G
#PBS -l walltime=45:00
#PBS -l nodes=1:ppn=1

cd $PBS_O_WORKDIR
mkdir -p logs/

# job code
echo Started: `date`
Rscript /home/richards/chen-yang.su/projects/richards/chen-yang.su/somalogic/associations_analysis.R $DATA $OUTCOME
echo Ended: `date` 

