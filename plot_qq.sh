#!/bin/bash

Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_A2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "infe A2 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_A3_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "infe A3 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_B2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "infe B2 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_C1_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "infe C1 age+sex+SampleGroup+ProcessTime+protlevel"

Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_non_infe_A2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "non_infe A2 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_non_infe_A3_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "non_infe A3 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_non_infe_B2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "non_infe B2 age+sex+SampleGroup+ProcessTime+protlevel"
Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_non_infe_C1_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "non_infe C1 age+sex+SampleGroup+ProcessTime+protlevel"
