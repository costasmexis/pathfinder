import pandas as pd
import numpy as np
import pickle
import requests
import re
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
    def __init__(self, pairs: pd.DataFrame, length=10):
        self.num_occurences = None # # number of times a metabolite appears on pairs dataset
        self.G = None # Graph structure
        self.pairs = pairs
        self.length = length

        self._get_number_of_occurences()
        self._pairs_preprocessing()

    ''' functions definitions '''

    def _pairs_preprocessing(self):
        # drop rows where RPAIRS_main is 0
        self.pairs = self.pairs[self.pairs['RPAIR_main'] != 0]
        

    def _get_number_of_occurences(self):
        self.num_occurences = pd.DataFrame(pd.DataFrame(pd.concat([self.pairs['source'], self.pairs['target']], axis=0)).value_counts())
    
    def create_graph(self, data: Data, pairs: pd.DataFrame):
        self.G = nx.from_pandas_edgelist(self.pairs, source='source', target='target', create_using=nx.Graph()) 
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

    def shortest_simple_paths(self, src, trg, weight=None):
        return list(islice(nx.shortest_simple_paths(self.G, source=src, target=trg, weight=weight), self.length))

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
    
    def constrained_shortest_path(self, src, trg, weight=None, rxn_doubling=True) -> list:
        paths = self.shortest_simple_paths(src, trg, weight=weight)
        if rxn_doubling:
            paths = [path for path in paths if not self._reaction_doubling(path)]
        
        # select path with max smiles similarity
        # count number of community changes in pathway
        smiles_sim, comm_changes = [], []
        for p in paths:
            if len(p) == 2: 
                print('***** Path with length 2', p, '*****')
                continue
            sum = 0
            chg = 0  # community changes per path
            compound_list = p
            for i in range(len(compound_list)-1):
                cpd_a = compound_list[i]
                cpd_b = compound_list[i+1]
                sum += self.G.edges[(cpd_a, cpd_b)]['smiles_similarity']
                if self.G.edges[(cpd_a, cpd_b)]['smiles_similarity'] == 0:
                    sum += self.G.edges[(cpd_b, cpd_a)]['smiles_similarity']
                if self.G.nodes[cpd_a]['community'] != self.G.nodes[cpd_b]['community']:
                    chg += 1

            smiles_sim.append(sum)
            comm_changes.append(chg)
        
        try:
            idx_smi = (smiles_sim.index(min(smiles_sim)))
            idx_com = (comm_changes.index(min(comm_changes)))
        except ValueError:
            idx_smi = None
            idx_com = None

        ''' check if no path is found '''
        if len(paths) == 0:
            # print(f'***** No path found between {src} and {trg} *****')
            # if no path is found, re-do constrained shortes path without chekcing for double reactions
            paths, idx_smi, idx_com = self.constrained_shortest_path(src, trg, weight=weight, rxn_doubling=False)

        return paths, idx_smi, idx_com
    
    def calculate_edge_mol_weight(self, data: Data):
        for edge in tqdm(self.G.edges()):
            a, b = edge[0], edge[1]
            if data.get_compound_by_id(a).is_cofactor or data.get_compound_by_id(b).is_cofactor:
                self.G.edges[(a, b)]['mol_weight'] = 10 # np.inf
            else:
                w_a = data.get_compound_by_id(a).mw
                w_b = data.get_compound_by_id(b).mw
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
            pred_path, idx_smi, idx_com = self.constrained_shortest_path(source, target, weight=method)
            try:
                pred_path = pred_path[idx_smi]
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

    ''' KEGG compound to BIGG compound '''
    def kegg_to_bigg_compound(self, kegg_id: str) -> str:
        response = requests.post(
            'http://bigg.ucsd.edu/advanced_search_external_id_results',
            data={'database_source': 'kegg.compound', 'query':kegg_id}
        )

        try:
            return re.search(r'/models/universal/metabolites/([^"]+)', response.text).group(1)
        except AttributeError:
            return 'Not found'

    ''' KEGG reaction to BIGG reaction '''
    def kegg_to_bigg_reaction(self, kegg_id: str) -> str:
        response = requests.post(
            'http://bigg.ucsd.edu/advanced_search_external_id_results',
            data={'database_source': 'kegg.reaction', 'query':kegg_id}
        )

        try:
            return re.search(r'/models/universal/reactions/([^"]+)', response.text).group(1)
        except AttributeError:
            return 'Not found'

    ''' check if a node exist in networkx graph'''
    def node_exists(self, node):
        return node in self.G.nodes()
    
    def get_edges(self):
        return self.G.edges(data=True)
    
    ''' get reaction by compounds '''
    def get_reaction_by_compounds(self, cpd_a, cpd_b):
        rxns = self.pairs[(self.pairs['source'] == cpd_a) & (self.pairs['target'] == cpd_b)]['Reaction'].values
        if len(rxns) == 0:
            rxns = self.pairs[(self.pairs['source'] == cpd_b) & (self.pairs['target'] == cpd_a)]['Reaction'].values
        return rxns
        