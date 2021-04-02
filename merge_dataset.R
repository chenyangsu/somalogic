#!/usr/bin/Rscript

######################################## Description ########################################
# This script merges the SomaLogic proteins dataset with another dataset containing patient information
# to form the infectious and non-infectious datasets. 
#############################################################################################
library(SomaDataIO)
library(dplyr)  
library(ggplot2)
library(readr)  # for read_tsv()
library(data.table)
library(readxl)
library(writexl)  # save in excel file


args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

soma_data = args[1]                                               # [normalized, unnormalized]: whether to use normalized or unnormalized SomaLogic dataset
nat_log_transf = args[2]                                          # [TRUE, FALSE]: whether to natural log transform proteins or not
standardize = args[3]                                             # [TRUE, FALSE]: whether to standardize proteins to mean 0 and std 1
remove_outliers = args[4]                                         # [TRUE, FALSE]: whether to remove outliers (protein values with std > 3 or std < -3)

setwd("/home/richards/chen-yang.su/projects/richards/chen-yang.su/somalogic")


if (soma_data == "normalized") {  # normalized SomaLogic data
  my_adat <- read.adat("./data/McGill-Richards-C-19 - SomaScan Data/MCG-200150/SS-200150_v4_ACDPlasma.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.medNormRefSMP.adat")

} else if (soma_data == "unnormalized") {  # unnormalized SomaLogic data
  my_adat <- read.adat("./data/McGill-Richards-C-19 - SomaScan Data/MCG-200150/SS-200150_v4_ACDPlasma.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.adat")
} else {
  stop("Invalid SomaLogic dataset. Options: {normalized/unnormalized}")
}


infe_dat <- read_tsv("./data/Somalogic_Inf_417_A3.txt")  # subset data for infected group (symptom onset day < 14)
non_infe_dat <- read_tsv("./data/Somalogic_NonInf_219_A3.txt")  # subset data for cleared group (symptom onset day > 31).


if (nat_log_transf == "TRUE") {  # Log Tranform with Base e
  my_adat <- log(my_adat)

} else if (nat_log_transf != "FALSE") {
  stop("Invalid transformation step. Options {TRUE/FALSE}")
}


length(getFeatureNames(my_adat))  # 5284 proteins ()

prot_list <- getFeatureNames(my_adat)  # get list of proteins
prot_list <- as.character(prot_list)  # convert to character

keep_cols <- c("SubjectID", "anonymized_patient_id", "PlateId", "SampleGroup", "sex", "age_at_diagnosis",
               "Days_symptom_update3", "ProcessTime", "A2", "A3", "B2", "C1")  # relevant columns that we want to keep

infe <- infe_dat[, keep_cols]  # select columns relevant to us
non_infe <- non_infe_dat[, keep_cols]

common_col = intersect(colnames(infe), colnames(my_adat))  # common col names except for "SubjectID" which is used for merge(,, by="SubjectID)
common_col <- common_col[!(common_col %in% c("SubjectID"))]

my_adat <- my_adat[, !(names(my_adat) %in% common_col)]  # drop common columns from my_adat (since SampleGroup=NAs, PlateID will be duplicated)
infe <- merge(infe, my_adat, by="SubjectID")
non_infe <- merge(non_infe, my_adat, by="SubjectID")


########## Standardization and Outlier removal ##########
##### standardization
if (standardize == "TRUE") {
  infe[, prot_list] <- scale(infe[, prot_list])  # standardize (mean=0, sd=1) each protein column
  non_infe[, prot_list] <- scale(non_infe[, prot_list])  # standardize (mean=0, sd=1) each protein column

  if (remove_outliers == "TRUE") {
    ##### remove outliers
    infe[, prot_list][infe[, prot_list] > 3] <- NA  # all protein values greater than 3sd from mean=0 is set to NA
    non_infe[, prot_list][non_infe[, prot_list] > 3] <- NA  # all protein values greater than 3sd from mean=0 is set to NA

  } else if (remove_outliers != "FALSE") {
    stop("Invalid option for removing outliers.")
  }

} else if (standardize != "FALSE") {
  stop("Invalid option")
}


## check that we get mean of 0 and sd of 1
# colMeans(infe[,prot_list])  # faster version of apply(scaled.dat, 2, mean)
# apply(infe[,prot_list], 2, sd)  # get sd of each column


infe_filename <- paste("./results/datasets/", "infe_417", "-soma_data=", soma_data, "-nat_log_tranf=", nat_log_transf, "-standardize=", standardize, "-remove_outliers=", remove_outliers,
                   ".csv", sep="")
print(infe_filename)

non_infe_filename <- paste("./results/datasets/", "non_infe_219", "-soma_data=", soma_data, "-nat_log_tranf=", nat_log_transf, "-standardize=", standardize, "-remove_outliers=", remove_outliers,
                                                              ".csv", sep="")
print(non_infe_filename)

write.csv(infe, infe_filename, row.names = FALSE)
write.csv(non_infe, non_infe_filename, row.names = FALSE)

