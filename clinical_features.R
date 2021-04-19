#!/usr/bin/Rscript

######################################## Description ########################################
# This script merges the 7 clinical variables corresponding to comorbidities which are required
# for the sensitivity analysis where we add these 7 features with age, sex, processtime, samplegroup + proteins
# to see whether it performs better than just age, sex, processtime, samplegroup + proteins

# com_diabetes: Diabetes - "0 No, 1 Yes, -1 Don't know"
# com_chronic_pulm: Chronic obstructive pulmonary disease (COPD) - "0 No, 1 Yes, -1 Don't know"
# com_chronic_kidney: Chronic kidney disease - "0 No, 1 Yes, -1 Don't know"
# com_heart_failure: Congestive heart failure - "0 No, 1 Yes, -1 Don't know" 
# com_hypertension: Hypertension - "0 No, 1 Yes, -1 Don't know"
# com_liver: Liver Disease - "0 No, 1 Yes, -1 Don't know"
# smoking: Smoking status - "0 Smoker, 1 Ex smoker, 2 Never smoker, -1 Don't know, -3 Prefer not to answer"

# Data dictionary: https://docs.google.com/spreadsheets/d/1hwBeqckB3_qC8nnavT0kLLntOh3GrmWRJQHeO9zwG8w/edit#gid=665246845
### Note: in data dictionary, com_diabetes doesn't exist
#############################################################################################

library(rio)

dat <- readRDS("/scratch/richards/tomoko.nakanishi/09.COVID19/05.BQC/BQC_phenotype/basic_JGH_CHUM_20210413.rds")
file <- paste("basic_JGH_CHUM_20210413.csv")
write.csv(dat, file, row.names = FALSE)

