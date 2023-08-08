import pandas as pd
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
from data import Data
import ast
from itertools import islice
from itertools import chain
from rdkit import Chem
from rdkit import DataStructs
from networkx.algorithms.community import greedy_modularity_communities

class Graph:
    def __init__(self, pairs: pd.DataFrame):
        self.num_occurences = None # # number of times a metabolite appears on pairs dataset
        self.G = None # Graph structure
        self.pairs = pairs

        self._get_number_of_occurences()

    # functions definitions
    def _get_number_of_occurences(self):
        self.num_occurences = pd.DataFrame(pd.DataFrame(pd.concat([self.pairs['source'], self.pairs['target']], axis=0)).value_counts())

    # main function to create graph    
    def create_graph(self, data: Data, pairs: pd.DataFrame):
        self.G = nx.from_pandas_edgelist(self.pairs, source='source', target='target', 
                                         create_using=nx.Graph()) 
        self_loops = list(nx.selfloop_edges(self.G))
        self.G.remove_edges_from(self_loops)
        print('# nodes:', self.G.number_of_nodes(), "\n# edges:", self.G.number_of_edges())

        # set node attributes
        community_df = self._get_community()
        for node in tqdm(self.G.nodes()):
            self.G.nodes[node]['mw'] = data.get_compound_by_id(node).mw
            self.G.nodes[node]['is_toxical'] = data.get_compound_by_id(node).is_toxic
            self.G.nodes[node]['is_cofactor'] = data.get_compound_by_id(node).is_cofactor
            self.G.nodes[node]['num_occurences'] = self.num_occurences.loc[node][0][0]
            self.G.nodes[node]['community'] = community_df[community_df['compound'] == node]['community'].values[0]
    
    # transform the community detection into a DataFrame
    def _get_community(self) -> pd.DataFrame:
        communities = self._community_detection()
        data = [{'community': idx, 'compound': cpd} for idx, community in enumerate(communities) for cpd in community]    
        return pd.DataFrame(data)

    # create or load saved communities (community detection)
    def _community_detection(self, weight=None):
        # check if file exists in folder
        try:
            with open('./data/communities.pkl', 'rb') as f:
                communities = pickle.load(f)
                return communities
        except FileNotFoundError:
            communities = greedy_modularity_communities(self.G, weight=weight)
            with open('./data/communities.pkl', 'wb') as f:
                pickle.dump(communities, f)
            return communities

    def shortest_simple_paths(self, src, trg, weight=None, length=10):
        return list(islice(nx.shortest_simple_paths(self.G, source=src, target=trg, weight=weight), length))

    # def to check if we are passing from a reaction twice
    def _reaction_doubling(self, path: list) -> bool:
        rxns_list = []
        for i in range(len(path)-1):
            cpd_a = path[i]
            cpd_b = path[i+1]
            rxns_list.append(self.pairs[(self.pairs['source'] == cpd_a) & (self.pairs['target'] == cpd_b)]['Reaction'].values)
            rxns_list.append(self.pairs[(self.pairs['source'] == cpd_b) & (self.pairs['target'] == cpd_a)]['Reaction'].values)
        
        rxns_list = [arr for arr in rxns_list if arr.any()]
        rxns_list = list(chain.from_iterable(rxns_list))
        return len(rxns_list) != len(set(rxns_list))
    
    def constrained_shortest_path(self, src, trg, weight=None) -> list:
        paths = self.shortest_simple_paths(src, trg, weight=weight)
        paths = [path for path in paths if not self._reaction_doubling(path)]
        
        # select path with max smiles similarity
        smiles_sim = []
        for p in paths:
            sum = 0
            compound_list = p
            for i in range(len(compound_list)-1):
                cpd_a = compound_list[i]
                cpd_b = compound_list[i+1]
                sum += self.G.edges[(cpd_a, cpd_b)]['smiles_similarity']
                if self.G.edges[(cpd_a, cpd_b)]['smiles_similarity'] == 0:
                    sum += self.G.edges[(cpd_b, cpd_a)]['smiles_similarity']
            
            smiles_sim.append(sum)
        
        try:
            idx = (smiles_sim.index(min(smiles_sim)))
        except ValueError:
            idx = None

        return paths, idx
    
    # function to check if cofactor
    def _check_for_cofactor(self, cpd) -> bool:
        return self.G.nodes[cpd]['is_cofactor']

    def calculate_edge_mol_weight(self, data: Data, elim_cofacs=True) -> None:
        for edge in tqdm(self.G.edges()):
            a, b = edge[0], edge[1]
            if elim_cofacs and (self._check_for_cofactor(a) or self._check_for_cofactor(b)):
                self.G.edges[(a, b)]['mol_weight'] = np.inf
            else:
                w_a = self.G.nodes[a]['mw']
                w_b = self.G.nodes[b]['mw']
                w = (np.abs(w_a - w_b) / (w_a + w_b + 1e-6))
                self.G.edges[(a, b)]['mol_weight'] = w

    def calculate_smiles_similarity(self, data: Data):
        for edge in tqdm(self.G.edges()):
            a, b = edge[0], edge[1]
            smiles1 = data.get_compound_by_id(a).smiles
            smiles2 = data.get_compound_by_id(b).smiles
            ms = [Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)]
            fs = [Chem.RDKFingerprint(x) for x in ms]
            s = DataStructs.FingerprintSimilarity(fs[0], fs[1])
            self.G.edges[(a, b)]['smiles_similarity'] = 1-s

    def validate(self, test_cases: pd.DataFrame, method: str):
        correct_pathways = []
        paths = []
        for row in tqdm(range(len(test_cases))):
            source = test_cases['source'].iloc[row]
            target = test_cases['target'].iloc[row]
            pred_path, idx = self.constrained_shortest_path(source, target, weight=method)
            try:
                pred_path = pred_path[idx]
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
