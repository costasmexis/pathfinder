import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from graph import Graph
from compound import Compound
from reaction import Reaction
from data import Data
import ast

def validate(test_cases: pd.DataFrame, G: Graph, method: str):
    correct_pathways = []
    paths = []
    for row in tqdm(range(len(test_cases))):
        source = test_cases['source'].iloc[row]
        target = test_cases['target'].iloc[row]
        try:
            pred_path, idx_smi, idx_com = G.constrained_shortest_path(source, target, weight=method)
            pred_path = pred_path[idx_smi[0]]
        except nx.NodeNotFound:
            print(f'***** Node not found for {source} or {target} *****')
            pred_path, idx_smi, idx_com = [], None, None
        except TypeError:
            pass
        
        correct_pathways.append((pred_path == test_cases['paths_list'].iloc[row]))
        paths.append(pred_path)
    
    print(f'Correct pathway predictions: {correct_pathways.count(True)}')
    print(f'Correct pathway predictions (%): {100 * correct_pathways.count(True) / len(correct_pathways)}')

    # return the DataFrame with the resulted pathways and correct or not
    paths = pd.DataFrame([str(p) for p in paths], columns=['Pathway'])
    paths['Pathway']  = paths['Pathway'].apply(lambda x: ast.literal_eval(x))
    paths['Correct'] = correct_pathways
    return paths

def create_compound(data: Data, cpds: pd.DataFrame, cofactors: pd.DataFrame):
    for index, row in cpds.iterrows():
        entry = row['Entry']
        name = row['Names']
        formula = row['Formula']
        mw = row['mol_weight']
        smiles = row['SMILES']
        is_cofactor = row['Entry'] in cofactors['Entry'].values
        is_polymer = row['polymer']

        compound = Compound(entry, name, formula, mw, smiles, is_cofactor, is_polymer)
        data.add_element('compound', compound)
    return data
    
def create_reaction(data: Data, rxns: pd.DataFrame):
    for index, row in rxns.iterrows():
        entry = row['Entry']
        name = row['Names']
        compounds = row['Compound']
        enzyme = row['EC Number']
        equation = row['Equation']

        reaction = Reaction(entry, name, compounds, enzyme, equation)
        data.add_element('reaction', reaction)
    return data

# Given 2 lists, check if the elements of the first list are in the second list
def check_list(l1: list, l2: list) -> bool:
    for i in l1:
        if i not in l2:
            return False
    return True

