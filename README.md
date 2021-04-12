# SomaLogic 

## Installation
Install Python 

    ```
    sudo apt update -y
    sudo apt install python3.7
    ```
 
### Python Environment
1. Create python virtual environment with `virtualenv` and install pip dependencies:
    ```
    mkdir -p $HOME/envs  # make envs directory and skip if already exists
    virtualenv -p /usr/bin/python3.7 $HOME/envs/somalogic  # create virtual environment with python 3.7
    source $HOME/envs/somalogic/bin/activate
    cd $HOME/PycharmProjects/somalogic
    pip install -r requirements.txt
    ```
   
## Intial Analysis
### Basic SomaLogic Data Download and Directory Setup

1. Set up directory
    ```
    mkdir -p somalogic  # -p flag allows no error returned if directory exists
    cd somalogic
    mkdir -p src data doc results ext
    ```

2. Download raw SomaLogic zip file from Hydra at `~/projects/richards/restricted/bqc19/data/somalogic/20201120/McGill-Richards-C-19-SomaScan-Data.zip` to your own directory
    ```
    # Download to local computer. While on local machine, run the following:
    cd data  # switch to data folder where file should be downloaded
    scp -r chen-yang.su@hydra.ladydavis.ca:~/projects/richards/restricted/bqc19/data/somalogic/20201120/McGill-Richards-C-19-SomaScan-Data.zip /mnt/c/Users/user/PyCharmProjects/somalogic/data
    unzip McGill-Richards-C-19-SomaScan-Data.zip  # unzip the file
    rm McGill-Richards-C-19-SomaScan-Data.zip  # remove the zip file
    ```

3. Unzip all subdirectories.
    ```
    cd 'McGill-Richards-C-19 - SomaScan Data'
    unzip MCG-200150.zip -d ./MCG-200150  # unzip file to location specified
    unzip SomaDataIO_3_1_0_and_pdf.zip -d ./SomaDataIO_3_1_0_and_pdf  
    rm MCG-200150.zip
    rm SomaDataIO_3_1_0_and_pdf.zip
    ```

4. Unzip `tar.gz` file in `.../SomaDataIO_3_1_0_and_pdf`. This will produce a folder `SomaDataIO` 
    ```
    cd SomaDataIO_3_1_0_and_pdf
    tar -xf SomaDataIO_3.1.0.tar.gz  # -x extract, f filename
    ```

5. Follow installation instructions in ./SomaDataIO/README.md file to install SomaLogic software and dependencies for R.

6. Read relevant pdf files to familiarize with SomaLogic data. `Filename (directory location)`
  - `SSM-00060 - Rev 1.0 - Data Standardization and File Specification Technical Note.pdf` (McGill-Richards-C-19 - SomaScan Data)
  - `SS-200150_SQS.pdf` (MCG-200150)
  - `SomaDataIO_3.1.0.pdf` (SomaDataIO_3_1_0_and_pdf)

### Association Analysis (in R)
- Note: SomaLogic provides us with two `.adat` files containing the protein levels. We are working with the **SomaLogic Normalized protein dataset**.

    - "Normalized" dataset: `SS-200150_v4_ACDPlasma.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.medNormRefSMP.adat`
    - "Unnormalized" dataset: `SS-200150_v4_ACDPlasma.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.adat`

Here, our goal is to run logistic regression in R while on the Hydra cluster. The output of this analysis will be a `.xlsx` file of 
association results with proteins sorted by increasing p values, odds ratios, and confidence intervals for each protein.
Furthermore, we will generate qq plots of the p values.

1. Create your infectious and non-infectious `.csv` file datasets by merging sample data with SomaLogic protein levels. 

    - To form one dataset, use `merge_dataset.R` with the following arguments:
    
        i. args[1] - whether to use normalized or unnormalized SomaLogic dataset
      
        ii. args[2] - whether to natural log transform proteins or not

        iii. args[3] - whether to standardize proteins to mean 0 and std 1
      
        iv. args[4] - whether to remove outliers (protein values with std > 3 or std < -3)
    ```
    Rscript merge_dataset.R {normalized/unnormalized} {TRUE/FALSE} {TRUE/FALSE} {TRUE/FALSE} 
   
    # To use the normalized SomaLogic dataset with proteins natural log transformed, unstandardized, and with no outliers removed
    Rscript merge_dataset.R normalized TRUE FALSE FALSE
    ```
    This automatically creates the infectious and non-infectious dataset.
    
   - To form all possible datasets
   ```
   bash merge_dataset.sh
   ```
   which will create 8 `.csv` file datasets (4 normalized, 4 unnormalized) for each of the infectious and non-infectious data for a total of 16 datasets
   stored in `./results/datasets/`   
   
2. Next, set up directory for saving results of association analysis
    ```
    cd results
    mkdir -p all_proteins
    cd all_proteins
    mkdir -p age+sex+SampleGroup+ProcessTime+protlevel
    cd age+sex+SampleGroup+ProcessTime+protlevel
    mkdir -p pvalues qqplots  # for storing pvalues text files and qq plot images, respectively
    ```   
   
3. Run association analyses for each dataset and each outcome. This will generate two files
    - `somalogic/results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/_{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+SampleGroup+ProcessTime+protlevel_Analysis=all_proteins.xlsx`
    which contains the odds ratio, confidence intervals, and p values. 
    - `somalogic/results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt` which takes the second to last
    column of the `.xlsx` file and stores it in a text file for later use in generating the QQ plots.
   ```
    Rscript associations_analysis.R {infe/non_infe} {A2/A3/B2/C1}
    # example run for infectious dataset and A2 outcome: Rscript associations_analysis.R infe A2
    ``` 
   - To launch jobs on the Hydra cluster:
    ```
    qsub -v "DATA=infe,OUTCOME=A2" launch_associations.sh  # for regression with infectious data on the A2 outcome
    ``` 
   
4. In the `ext` directory, clone qq.plink repo for plotting qq plots. 
    ```
    cd ext
    git clone https://github.com/vforget/qq.plink 
    ```
   
5. Generate qq plots with qq.plink
    ```
    Rscript ./ext/qq.plink/qq.plink.R $pvalue.txt$ "some title"  
   
    # example for p value results from infe dataset with A2 outcome
    Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_A2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt "infe A2 age+sex+SampleGroup+ProcessTime+protlevel"
    ```
   where the first argument `$pvalue.txt$` should be replaced with the saved p value text file and the second argument `"some title"` is the title for the plot
   which should be surrounded in double quotes "". The output will be `./results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues/_infe_A2_LR_age+sex+SampleGroup+ProcessTime+protlevel_pvalues.txt.qq.png`
   
   - To run all analyses at once, launch `plot_qq.sh`. This will generate QQ plots for all combinations `{infe/non_infe} {A2/A3/B2/C1}`
    ```
    bash plot_qq.sh
    ```
   Finally, move all QQ plots to `qqplots` folder
    ```
    cd results/all_proteins/age+sex+SampleGroup+ProcessTime+protlevel/pvalues
    mv *.qq.png ../qqplots
    ```

## Training
- Our original list contains 5284 SomaLogic SOMAmers. 
For this analysis, we remove `NoneX.*`, `NonHuman`, `Spuriomer`, `HybControlElution`, `NonBiotin`, `NonCleavable` 
from the list to get a total of **4984 proteins**. This text file of proteins is stored as `somalogic/data/Somalogic_list_QC1.txt`

### Regularized Association Analysis with different covariates
<!-- # Used only if want to also model FDR proteins

1. Form datasets. This will create a dataset `somalogic/results/datasets/{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+SampleGroup+ProcessTime+protlevel.csv`
   with each row being a sample and the columns: `age_at_diagnosis, sex, protein_1, protein_2, ..., protein_5284`.
   The protein columns are sorted by increasing p value so `protein_1` has the lowest p value and `protein_5284` has the highest
   p value (close to 1)
    ```
    python make_dataset.py --data {infe/non_infe} --outcome {A2/A3/B2/C1}
   
    # example run
    python make_dataset.py --data infe --outcome A2
    ```
-->

1. Run models. This will perform a hyperparameter search using Stratified 5 fold cross-validation and use the best 
hyperparameter to train on the entire training set and save the final model.
    ```
    python models.py --soma_data {normalized/unnormalized} --nat_log_transf {True/False} --standardize {True/False} --data {infe/non_infe} --outcome {A2/A3/B2/C1} --model_type {lasso/elasticnet} --params_search {True/False}
  
    # example run 
    python models.py --soma_data normalized --nat_log_transf True --standardize True --data infe --outcome A2 --model_type lasso --params_search True
    ```
    Note: after running once with `--params_search True`, the model parameters will be saved as a `.pkl` file such as 
    `somalogic/results/models/lasso-soma_data=normalized-nat_log_transf=False-standardize=False_infe_A2_results.pkl` if the above command were run. Since the parameters are saved already,
    subsequent runs should set `--params_search False` since `models.py` will directly load this `.pkl` file.

<!-- 
### Directory Information
- `src`: where scripts are located
- `data`: raw data and example/toy data
- `doc`:  manuscript files, source code documentation.
- `ext`: software from external sources
- `results`: cleaned datasets, figures and tables
-->

## Testing the Model
1. Merge your SomaLogic Normalized dataset with patient information and name this file `test.csv`. 
Do not perform any preprocessing on this dataset. i.e. no log transformation, scaling, outlier removal of any sort. 
All protein levels should be the raw values in the original SomaLogic Normalized dataset.

1. Define A2 and A3 outcomes within this dataset

1. Store your SomaLogic Normalized dataset under `somalogic/results/datasets/test`.
    ```
    cd somalogic
    mkdir -p results/datasets
    mkdir -p results/datasets/test
    cd results/datasets/test
    # store test.csv dataset in results/datasets/test directory
    ```
1. Test model on A2 and A3 outcomes
    ```
    python test_models.py --soma_data normalized --nat_log_transf True --standardize True --data infe --outcome {A2/A3} --model_type lasso
    ```
