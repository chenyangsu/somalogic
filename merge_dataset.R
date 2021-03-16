#!/usr/bin/Rscript

########## Description ##########
# Reads in SomaLogic (Normalized) protein level data and merges it with sample information of preformed
infectious and non-infectious datasets which contain sample level information only
Next, this script performs the following transformations to the proteins:
# 1. Proteins are natural log (base e) transformed
# 2. Each protein is standardized to a mean of 0 and std of 1
# 3. Removal of outliers: any protein values above or below 3 std are removed.
Finally, datasets are saved as .csv files.
#################################

library(SomaDataIO)
library(dplyr)  
library(ggplot2)
library(readr)  # for read_tsv()
library(data.table)
library(readxl)
library(writexl)  # save in excel file

setwd("/home/richards/chen-yang.su/projects/richards/chen-yang.su/somalogic")

my_adat <- read.adat("./data/McGill-Richards-C-19 - SomaScan Data/MCG-200150/
                     SS-200150_v4_ACDPlasma.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.medNormRefSMP.adat")

infe_dat <- read_tsv("./data/Somalogic_Inf_418_A3.txt")  # subset data for infected group (symptom onset day < 14)
non_infe_dat <- read_tsv("./data/Somalogic_NonInf_220_A3.txt")  # subset data for cleared group (symptom onset day > 31).

my_adat <- log(my_adat)  # Log Tranform with Base e 

length(getFeatureNames(my_adat))  # 5284 proteins ()

prot_list <- getFeatureNames(my_adat)  # get list of proteins

prot_list <- as.character(prot_list)  # convert to character

keep_cols <- c("SubjectID", "anonymized_patient_id", "PlateId", "sex", "age_at_diagnosis",
               "Days_symptom_update3", "ProcessTime", "A2", "A3", "B2", "C1")  # relevant columns that we want to keep

infe <- infe_dat[, keep_cols]  # select columns relevant to us
non_infe <- non_infe_dat[, keep_cols]

common_col = intersect(colnames(infe), colnames(my_adat))  # common col names except for "SubjectID" which is used for merge(,, by="SubjectID)
common_col <- common_col[!(common_col %in% c("SubjectID"))]

infe <- infe[, !(names(infe) %in% common_col)]  # drop common columns from samples_info so merging won't have duplicates
infe <- merge(infe, my_adat, by="SubjectID")  

non_infe <- non_infe[, !(names(non_infe) %in% common_col)] 
non_infe <- merge(non_infe, my_adat, by="SubjectID")  


infe[, prot_list] <- scale(infe[, prot_list])  # standardize (mean=0, sd=1) each protein column
non_infe[, prot_list] <- scale(non_infe[, prot_list])  # standardize (mean=0, sd=1) each protein column

# check that we get mean of 0 and sd of 1
# colMeans(infe[,prot_list])  # faster version of apply(scaled.dat, 2, mean)
# apply(infe[,prot_list], 2, sd)  # get sd of each column 


infe[, prot_list][infe[, prot_list] > 3] <- NA  # all protein values greater than 3sd from mean=0 is set to NA
non_infe[, prot_list][non_infe[, prot_list] > 3] <- NA  # all protein values greater than 3sd from mean=0 is set to NA


write.csv(infe, "./results/datasets/infe_418.csv", row.names = FALSE)
write.csv(non_infe, "./results/datasets/non_infe_418.csv", row.names = FALSE)

