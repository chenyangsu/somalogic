# SomaLogic 

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
We first run logistic regression in R while on the Hydra cluster using the file `lr_qq.R`. The output of this analysis will be a `.xlsx` file of 
association results with proteins sorted by increasing p values, odds ratios, and confidence intervals for each protein.
Furthermore, we will generate qq plots of the p values.

1. Create your infectious and non-infectious dataset by merging sample data with SomaLogic protein levels.
This will create two separate files `./results/datasets/infe_418.csv` and `./results/datasets/non_infe_418.csv`
    ```
    Rscript merge_dataset.R
    ```
2. Next, set up directory for saving results of association analysis
    ```
    cd results
    mkdir -p all_proteins
    cd all_proteins
    mkdir -p age+sex+protlevel
    cd age+sex+protlevel
    mkdir -p pvalues qqplots  # for storing pvalues text files and qq plot images, respectively
    ```   
3. Run association analyses for each dataset and each outcome. This will generate two files
    - `somalogic/results/all_proteins/age+sex+protlevel/_{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+protlevel_Analysis=all_proteins.xlsx`
    which contains the odds ratio, confidence intervals, and p values. 
    - `somalogic/results/all_proteins/age+sex+protlevel/pvalues/_{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+protlevel_pvalues.txt` which takes the second to last
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
    Rscript ./ext/qq.plink/qq.plink.R ./results/all_proteins/age+sex+protlevel/pvalues/_infe_A2_LR_age+sex+protlevel_pvalues.txt "infe A2 age+sex+protlevel"
    ```
   where the first argument `$pvalue.txt$` should be replaced with the saved p value text file and the second argument `"some title"` is the title for the plot
   which should be surrounded in double quotes "". The output will be `./results/all_proteins/age+sex+protlevel/pvalues/_infe_A2_LR_age+sex+protlevel_pvalues.txt.qq.png`
   
   - To run all analyses at once, launch `plot_qq.sh`. This will generate QQ plots for all combinations `{infe/non_infe} {A2/A3/B2/C1}`
    ```
    bash plot_qq.sh
    ```
   Finally, move all QQ plots to `qqplots` folder
    ```
    cd results/all_proteins/age+sex+protlevel/pvalues
    mv *.qq.png ../qqplots
    ```

### Regularized Association Analysis with different covariates
1. Form datasets. This will create a dataset `somalogic/results/datasets/{infe/non_infe}_{A2/A3/B2/C1}_LR_age+sex+protlevel.csv`
   with each row being a sample and the columns: `age_at_diagnosis, sex, protein_1, protein_2, ..., protein_5284`.
   The protein columns are sorted by increasing p value so `protein_1` has the lowest p value and `protein_5284` has the highest
   p value (close to 1)
    ```
    python make_dataset.py {infe/non_infe} {A2/A3/B2/C1}
    ```
2. Run models
    ```
    python models.py
    ```
### Directory Setup (High-Level Overview of Important Folders)
- data
    - McGill-Richards-C-19 - SomaScan Data  `Raw SomaLogic Data Directory`
        - MCG-200150
        - SomaDataIO_3_1_0_and_pdf
            - SomaDataIO


### Directory Information
- `src`: where scripts are located
- `data`: raw data and example/toy data
- `doc`:  manuscript files, source code documentation.
- `ext`: software from external sources
- `results`: cleaned datasets, figures and tables


#######
Modify `\src\lr.R` and then to launch jobs (while in`src` folder), 

```
qsub infe_{A2/B2/C1}.sh
```

Will output results to the folder `somalogic/results/`
######





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

## How to run
1. Create your data sets which will be saved in `...\somalogic\results\datasets`
    ```
    python make_dataset.py --outcome A2
    python make_dataset.py --outcome B2
    python make_dataset.py --outcome C1
    ```


## How to run
- Single run:
  ``` 
  python run_{rac/rairl}_{classic/bandit/cbandit}_{train/eval}.py 
  ```

- Sweep: We will use Sweep for two purposes:
    1. Hyperparameter search via either random search or Bayesian optimization.
    2. After finding out hyperparameter, run it through multiple seeds with fixed hyperparameters.
  Both will be done with wandb's Sweep concatenated with slurm in CC clusters. For each purpose, we need to do the followings at the **project root** directory (`cd $HOME/PycharmProjects/RegAIRL`):
  1. Create directory `{sweep_name}` inside `sweeps`, e.g., `sweeps/{sweep_name}`. Naming convention for `sweep_name`:
      ```
      sweep_name = {rac/rairl}_{train/eval}_{bayes/random/grid}_{env_id}_{reg_type}_{yymmdd-version}
      ``` 
      We will use `train` for IRL and `eval` for Reward evaluation.  
      For example: `rac_train_bayes_Pendulum-v0_shannon_200730-v0`
  2. Create configuration file './sweep_config.py'. The `parameters` for `sweep_config` can be found in the parsers of your target `run` file. See [this link](https://docs.wandb.com/sweeps/configuration) for details.
  3. Initialize Sweep with `init_sweep.py` with the following command:
      ``` 
      python init_sweep.py sweeps/{sweep_name}
      python init_sweep.py sweeps/{sweep_name}/  # also works
      ```
     This will generate `sweeps/{sweep_name}/sweep_id` file.
  4. Launch agent with `launch_agent.sh` with one of the following commands:
      ``` 
      sbatch {add additional sbatch flags if needed} launch_agent.sh sweeps/{sweep_name}
      sbatch {add additional sbatch flags if needed} launch_agent.sh sweeps/{sweep_name}/  # also works
      sbatch {add additional sbatch flags if needed} launch_agent.sh {sweep_name}          # also works
      sbatch {add additional sbatch flags if needed} launch_agent.sh {sweep_name}/         # also works
      ```
     e.g., `sbatch --array=1-100 --partition=unkillable launch_agent.sh test`. This command will launch `wandb agent` command inside for each jobs. 

## Experiment results

- Results will be uploaded to wandb as well as cluster servers (`$PROJECT_RAIRL/results/RegAIRL`).

## Reference
- [@astooke/rlpyt](https://github.com/astooke/rlpyt)


### Directory Information
```somalogic
├── ext
│   ├── claudia_langenberg_code
│   └── qq.plink
├── data
│   └── McGill-Richards-C-19 - SomaScan Data
│       ├── MCG-200150
│       └── SomaDataIO_3_1_0_and_pdf
│           └── SomaDataIO
│               ├── R
│               ├── build
│               ├── data
│               ├── inst
│               │   ├── doc
│               │   ├── install
│               │   └── manual
│               ├── man
│               ├── tests
│               │   └── testthat
│               └── vignettes
├── doc
│   └── literature
├── results
│   ├── all_proteins
│   │   ├── RowCheck=PASS_age+sex+protlevel
│   │   │   ├── outliers_not_removed
│   │   │   └── remove_outliers
│   │   ├── UNNORMALIZED_age+sex+PC1-20
│   │   ├── UNNORMALIZED_age+sex+protlevel
│   │   │   ├── pvalues
│   │   │   └── qqplots
│   │   ├── age+sex+protlevel
│   │   │   ├── principal_components_FDRp
│   │   │   ├── pvalues
│   │   │   └── qqplots
│   │   ├── age+sex+protlevel+PC1-8
│   │   │   ├── pvalues
│   │   │   └── qqplots
│   │   ├── age+sex+protlevel+PlateId+SampleGroup+ProcessTime.hrs.-under-_24_hr
│   │   │   ├── 210119
│   │   │   ├── 210121
│   │   │   │   ├── pvalues
│   │   │   │   └── qqplot
│   │   │   └── bs()
│   │   │       ├── bsprotein1
│   │   │       │   ├── pvalues
│   │   │       │   └── qqplots
│   │   │       ├── bsprotein2
│   │   │       │   ├── pvalues
│   │   │       │   └── qqplots
│   │   │       └── randomized_qqplots
│   │   ├── age+sex+protlevel+age2
│   │   │   ├── p
│   │   │   └── qqplots
│   │   ├── age_over_60+sex+protlevel
│   │   │   ├── pvalues
│   │   │   └── qqplots
│   │   └── age_over_60+sex+protlevel+age2
│   │       └── pvalues
│   ├── crp
│   ├── datasets  # for experimental datasets 
│   ├── mbl2
│   ├── mr_significant_proteins
│   │   └── sirui_annotated
│   └── plots
│       └── infectious
│           ├── A2
│           │   ├── A2_boxplots
│           │   ├── A2_histogram
│           │   └── A2_spearman_correlation
│           │       ├── All
│           │       ├── UNNORMALIZED top 50 using normalized_a2_cases proteins
│           │       └── top50
│           ├── B2
│           │   ├── B2_boxplots
│           │   ├── B2_histogram
│           │   └── B2_spearman_correlation
│           │       └── top50
│           ├── C1
│           │   ├── C1_boxplots
│           │   ├── C1_histogram
│           │   └── C1_spearman_correlation
│           │       ├── top50
│           │       └── unnormalized top 50_with_normalized C1_proteins
│           └── spearman_abs_top50
└── src
    ├── bash
    │   ├── qqplots
    │   └── submit_jobs
    │       ├── hydra
    │       └── mila
    ├── python
    └── r
```