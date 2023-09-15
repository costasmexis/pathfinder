import pandas as pd
import numpy as np
import json
import networkx as nx
from tqdm import tqdm
from compound import Compound
from reaction import Reaction
from graph import Graph
from data import Data
from pathway import Pathway

# suppres rdkit warnings
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def smiles_to_fingerprint_sample(df: pd.DataFrame):
    smi_sample = df.iloc[123]['SMILES']
    print(smi_sample)

    mol = Chem.MolFromSmiles(smi_sample)

    fingerprint_rdk = Chem.RDKFingerprint(mol)
    print(">>> RDK Fingerprint = ", fingerprint_rdk)
    fingerprint_rdk_np = np.array(fingerprint_rdk)
    print(">>> RDK Fingerprint in numpy = ", fingerprint_rdk_np)
    print(">>> RDK Fingerprint in numpy shape = ", fingerprint_rdk_np.shape)

    print()

    # retrieving Morgan Fingerprint ----------------------------------------------- 
    fingerprint_morgan = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)
    print(">>> Morgan Fingerprint = ", fingerprint_morgan)

    fingerprint_morgan_np = np.array(fingerprint_morgan)
    print(">>> Morgan Fingerprint in numpy : ", fingerprint_morgan_np)
    print(">>> Morgan Fingerprint in numpy shape = ", fingerprint_morgan_np.shape)

def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = Chem.RDKFingerprint(mol)
    return np.array(fp)

def main():
        
    # read data from csv
    cpds = pd.read_csv('../GNN_toxic/data/raw/compounds_final.csv', index_col=0) # containing toxicity
    rxns = pd.read_csv('data/reactions_final.csv', index_col=0)
    pairs = pd.read_csv('data/pairs_final_RPAIRS.csv', index_col=0)
    cofactors = pd.read_csv('data/original/cofactors_KEGG.csv')

    # keep rows with RPAIR_main != 2
    df_train = pairs[pairs['RPAIR_main'] != 2]
    df_test = pairs[pairs['RPAIR_main'] == 2]
    print(f'Shape of total dataset: {pairs.shape}')
    print(f'Shape of train dataset: {df_train.shape}')
    print(f'Shape of test dataset: {df_test.shape}')


    pairs_smiles_list = []
    for row in tqdm(range(len(pairs))):
        source = pairs.iloc[row]['source']
        source_smi = cpds[cpds['Entry'] == source]['SMILES'].values[0]

        target = pairs.iloc[row]['target']
        target_smi = cpds[cpds['Entry'] == target]['SMILES'].values[0]

        source_smi = get_fingerprint(source_smi)
        target_smi = get_fingerprint(target_smi)
        
        pair_smi = source_smi + target_smi
        pairs_smiles_list.append(pair_smi)

    pairs_smiles_df = pd.DataFrame(pairs_smiles_list)

    pairs_smiles_df['RPAIR_main'] = pairs['RPAIR_main'].values

    # save pairs_smiles_df to csv
    pairs_smiles_df.to_csv('data/pairs_final_RPAIRS_smiles.csv')

if __name__ == '__main__':
    main()
