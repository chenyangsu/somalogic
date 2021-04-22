#!/usr/bin/Rscript

########## Description ##########
# Runs one Logistic regression model per protein with the formula = outcome~age+sex+protein to get results for each protein (sorted by decreasing p value, lowest p to highest p): 
# "odds ratio, upper confidence interval, lower confidence interval, p value, fdr p value"
# This file is saved as an .xlsx file.
# Finally, it generates a text file with a single column corresponding to the p values (for use in qq plotting with qq.plink)
#################################

library(SomaDataIO)
library(dplyr)  
library(ggplot2)
library(readr)  # for read_tsv()
library(data.table)
library(readxl)
library(writexl)  # save in excel file


########## Logistic Regression ##########

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

data = args[1]                                                # [infe, non_infe]
outcome = args[2]                                             # [A2, A3, B2, C1]
# data = "infe"
# outcome = "A2"

setwd("/home/richards/chen-yang.su/projects/richards/chen-yang.su/somalogic")

# Start running from here for different outcomes A2, B2, C1
df <- data.frame()  # MUST run this first!! to make sure dataframe not already filled with a protein
i <- 0
formula <- A2 ~ age_at_diagnosis + sex + protein

if (data == "infe") {  # set dat
  dat <- read.csv(file = "./results/datasets/infe_417-soma_data=normalized-nat_log_tranf=TRUE-standardize=TRUE-remove_outliers=FALSE.csv")
  
} else if (data == "non_infe") {
  dat <- read.csv(file = "./results/datasets/non_infe_219.csv")
  
} else {
  stop("The data inputted is incorrect. Select one of {infe, non_infe}.") 
}
# print(table(dat$A2))
# print(table(dat$A3))
prot_list = readLines("Somalogic_list_QC1.txt")

for (prot in prot_list) {
  print(prot)
  i <- i + 1
  print(i / length(prot_list) * 100)  # percent done!
  
  protein <- dat[ , prot]  # KEY STEP or get error                               
  
  if (outcome == "A2") {
    formula <- A2 ~ age_at_diagnosis + sex + SampleGroup + ProcessTime + protein 
    
  } else if (outcome == "A3") {
    formula <- A3 ~ age_at_diagnosis + sex + SampleGroup + ProcessTime + protein 
    
  } else if (outcome == "B2") {
    formula <- B2 ~ age_at_diagnosis + sex + SampleGroup + ProcessTime + protein
    
  } else if (outcome == "C1") {
    formula <- C1 ~ age_at_diagnosis + sex + SampleGroup + ProcessTime + protein 
    
  } else{
    stop("The outcome inputted is incorrect. Select one of {A2, A3, B2, C1}.")
    
  }
  
  # logit removes NA observations by default
  logit <- glm(formula, data = dat, family = 'binomial')  # logistic regression                         
  sum_stat <- summary(logit)
  
  a <- exp(cbind(OR = coef(logit), confint(logit)))  # transform log(odds) to odds
  
  p <-sum_stat$coefficients[6, 4]  # get p value of protein
  or <- a[6, 1]  # get odds ratio of protein
  ci_low <- a[6, 2]  # get lower confidence interval of protein
  ci_high <- a[6, 3]  # get higher confidence interval of protein
  
  df1 <- data.frame(Protein = c(prot),
                    OR = or,
                    CIL = ci_low, 
                    CUL = ci_high,
                    P = p)
  df <- rbind(df, df1) 
}

df <- df[order(df$P), ]  # sort by increasing p value (smallest to largest p)

df$FDRp <- p.adjust(df$P, method = "fdr", n = length(prot_list))  # Create column with FDR corrected p value (Benjamini-Hochberg method)
stop()
file_name <- paste("./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/", data, outcome, "LR",            
                   "age+sex+SampleGroup+ProcessTime+protlevel", "Analysis=all_proteins.xlsx", sep = "_")  # concatenates arguments by separator

message("Saving results to file = ", file_name)

write_xlsx(df, file_name)

message("Done!")


########## Generate text file of p values #########
# Produce text file with a single column corresponding to the p values  (for use in qq plotting with qq.plink)

# write column of p values into text file with col name "P" (for qq.link plotting)
df <- read_excel(file_name)  

p <- df[, ncol(df)-1]  # get second to last column which stores p values (the last column are the FDR p values)

names(p) <- "P"  # rename column head (required to be name "P" for use with qq.plink)

output_file <- paste("./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/", data, outcome, "LR",            
                     "age+sex+SampleGroup+ProcessTime+protlevel", "pvalues.txt", sep = "_") 

message("Writing P values to file = ", output_file)

write.table(p, output_file, sep="\n", row.names=FALSE)

message("Done!")


