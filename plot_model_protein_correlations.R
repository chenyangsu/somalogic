# install.packages("ggcorrplot")
library(ggcorrplot)

# read in spearman correlation csv files saved using protein_correlations.ipynb
a2 <- read.csv(file = '../PycharmProjects/somalogic/results/datasets/infe_A2_lasso_proteins_spearman.csv')

a3 <- read.csv(file = '../PycharmProjects/somalogic/results/datasets/infe_A3_lasso_proteins_spearman.csv')

# read in raw protein values
a2_raw <- read.csv(file = '../PycharmProjects/somalogic/results/datasets/infe_A2_lasso_proteins_raw.csv')
a3_raw <- read.csv(file = '../PycharmProjects/somalogic/results/datasets/infe_A3_lasso_proteins_raw.csv')



a2_colnames <- colnames(a2)[2:length(a2)]  # get colnames starting at column 2 so ignore first column name which is X
a2_corr <- a2[, a2_colnames]  # get subset of original df starting from second column to end
rownames(a2_corr) <- a2_colnames  # rename the row indices with the column names


a3_colnames <- colnames(a3)[2:length(a3)]
a3_corr <- a3[, a3_colnames]
rownames(a3_corr) <- a3_colnames


a2_raw <- a2_raw[, a2_colnames]  # get subset of original df starting from second column to end
a2_p.mat <- cor_pmat(a2_raw, method="spearman")


a3_raw <- a3_raw[, a3_colnames]  # get subset of original df starting from second column to end
a3_p.mat <- cor_pmat(a3_raw, method="spearman`")


ggcorrplot(a2_corr, type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white")  # rotate x axis to 90 degrees
ggsave(filename="infe_A2_lasso_model_proteins_spearman_hierarchical_clustering.png",
       width=12, 
       height=12)

ggcorrplot(a3_corr, type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white")  # rotate x axis to 90 degrees
ggsave(filename="infe_A3_lasso_model_proteins_spearman_hierarchical_clustering.png",
       width=12, 
       height=12)
# rounding numbers to one decimal place3
# round(a3_corr, digits=1)

# ggcorrplot(round(a3_corr, digits=1), type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white", lab = TRUE, lab_size=2)  # rotate x axis to 90 degrees

ggcorrplot(a3_corr, type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white")  # rotate x axis to 90 degrees

ggcorrplot(a2_corr, type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white", p.mat=a2_p.mat, insig = "blank")  # rotate x axis to 90 degrees

ggcorrplot(a3_corr, type="lower", hc.order = TRUE, tl.srt = 90, outline.color="white", p.mat=a3_p.mat, insig = "blank")  # rotate x axis to 90 degrees
# type="lower" - only plot the lower triangle
# tl.srt = 90 - rotate x axis labels by 90 degrees so vertical
# outline.color="white" - set the outline of each box to white
# p.mat = a2_p.mat - Barring the no significant coefficient (adds correlation significance level)
# insig = "blank" - remove squares that are nonsignificant p values

# lab = TRUE - annotate the squares with the actual correlation values
# lab_size = 2 - set the size of the annotated values



