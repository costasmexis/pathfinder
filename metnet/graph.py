import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from data import Data
import ast
from itertools import islice
from itertools import chain
from rdkit import Chem
from rdkit import DataStructs

class Graph:
    def __init__(self, pairs: pd.DataFrame):
        self.num_occurences = None # # number of times a metabolite appears on pairs dataset
        self.G = None # Graph structure
        self.pairs = pairs

    def get_number_of_occurences(self, pairs):
        self.num_occurences = pd.DataFrame(pd.DataFrame(pd.concat([self.pairs['source'], self.pairs['target']], axis=0)).value_counts())
        
    def create_graph(self, data: Data, pairs: pd.DataFrame):
        self.G = nx.from_pandas_edgelist(self.pairs, source='source', target='target', 
                                         create_using=nx.Graph()) 
        self_loops = list(nx.selfloop_edges(self.G))
        self.G.remove_edges_from(self_loops)
        print('# nodes:', self.G.number_of_nodes(), "\n# edges:", self.G.number_of_edges())

        # set node attributes
        for node in tqdm(self.G.nodes()):
            self.G.nodes[node]['mw'] = data.get_compound_by_id(node).mw
            self.G.nodes[node]['is_toxical'] = data.get_compound_by_id(node).is_toxic
            self.G.nodes[node]['is_cofactor'] = data.get_compound_by_id(node).is_cofactor

    def shortest_simple_paths(self, src, trg, weight=None, length=10):
        return list(islice(nx.shortest_simple_paths(self.G, source=src, target=trg, weight=weight), length))

    def has_duplicates(self, lst) -> bool:
        return len(lst) != len(set(lst))
    
    def constrained_shortest_path(self, src, trg, weight=None) -> list:
        paths = self.shortest_simple_paths(src, trg, weight=weight)
        return paths
    
    def calculate_edge_mol_weight(self, data: Data):
        for edge in tqdm(self.G.edges()):
            a, b = edge[0], edge[1]
            if data.get_compound_by_id(a).is_cofactor or data.get_compound_by_id(b).is_cofactor:
                self.G.edges[(a, b)]['mol_weight'] = np.inf
            else:
                w_a = data.get_compound_by_id(a).mw
                w_b = data.get_compound_by_id(b).mw
                w = (np.abs(w_a - w_b) / (w_a + w_b + 1e-6))
                self.G.edges[(a, b)]['mol_weight'] = w


    def calculate_smiles_similarity(self, data: Data):
        for edge in tqdm(self.G.edges()):
            a, b = edge[0], edge[1]
            if data.get_compound_by_id(a).is_cofactor or data.get_compound_by_id(b).is_cofactor:
                self.G.edges[(a, b)]['mol_weight'] = np.inf
            else:
                smiles1 = data.get_compound_by_id(a).smiles
                smiles2 = data.get_compound_by_id(b).smiles
                ms = [Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)]
                fs = [Chem.RDKFingerprint(x) for x in ms]
                s = DataStructs.FingerprintSimilarity(fs[0], fs[1])
                self.G.edges[(a, b)]['smiles_similarity'] = 1-s

    def validate(self, test_cases: pd.DataFrame, method: str):
        correct_pathways = []
        paths = []
        for row in range(len(test_cases)):
            source = test_cases['source'].iloc[row]
            target = test_cases['target'].iloc[row]
            correct_pathways.append((list(nx.shortest_path(self.G, source, target, weight=method)) == test_cases['paths_list'].iloc[row]))
            paths.append(list(nx.shortest_path(self.G, source, target, weight=method)))

        print(f'Correct pathway predictions: {correct_pathways.count(True)}')
        print(f'Correct pathway predictions (%): {100 * correct_pathways.count(True) / len(correct_pathways)}')

        # return the DataFrame with the resulted pathways and correct or not
        paths = pd.DataFrame([str(p) for p in paths], columns=['Pathway'])
        paths['Pathway']  = paths['Pathway'].apply(lambda x: ast.literal_eval(x))
        paths['Correct'] = correct_pathways
        return paths