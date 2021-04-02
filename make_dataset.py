"""
Forms dataset with proteins sorted by increasing p value going from left to right
So, order of protein columns is from lowest p value to highest p values (left to right)
"""

import argparse
import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../somalogic

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='infe',
                        choices=['infe', 'non_infe'], help='The dataset to use')
    parser.add_argument('--outcome', type=str, default='A2',
                        choices=['A2', 'A3', 'B2', 'C1'], help='The COVID severity outcome')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    config = vars(parser.parse_args())

    data = config['data']
    outcome = config['outcome']
    if data == "infe":
        file_name = data + '_417.csv'
    elif data == "non_infe":
        file_name = data + '_219.csv'

    file = os.path.join(ROOT_DIR, 'results', 'datasets', file_name)
    df = pd.read_csv(file, low_memory=False)

    dataset = f'_{data}_{outcome}_LR_age+sex+SampleGroup+ProcessTime+protlevel_Analysis=all_proteins.xlsx'  # dataset from association analysis

    variables = 'age+sex+SampleGroup+ProcessTime+protlevel'

    DATA_DIR = os.path.join(ROOT_DIR, 'results', 'all_proteins', variables)
    file_path = os.path.join(DATA_DIR, dataset)

    results = pd.read_excel(file_path, engine='openpyxl')  # without engine, will not open

    proteins = list(results.iloc[:, 0])  # get list of proteins sorted by increasing p value

    col = ['age_at_diagnosis', 'sex', 'SampleGroup', 'ProcessTime', outcome] + proteins

    df = df[col]

    save_file = data + '_' + outcome + '_LR_age+sex+SampleGroup+ProcessTime+protlevel.csv'

    file_to_save = os.path.join(ROOT_DIR, 'results', 'datasets', save_file)

    df.to_csv(file_to_save, index=False)  # don't save row index
