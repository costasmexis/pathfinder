import pandas as pd
import numpy as np
import requests
import re

class Pathway:
    def __init__(self):
        self.source = None
        self.target = None
        self.graph = None
        
        self.weight = 'mol_weight'
        
        self.pred_paths = None
        self.idx_smi = None
        self.idx_com = None

        # Store signle paths (compound form, reaction form)
        self.path_compound = None
        self.path_reactions = None
    
    def initialize(self, source, target, graph):
        self.source = source
        self.target = target
        self.graph = graph

    def get_pathway(self):
        self.pred_paths, self.idx_smi, self.idx_com = self.graph.constrained_shortest_path(self.source, self.target, weight=self.weight)

    def print_pathway(self, with_reactions=False, to_bigg=False):
        print(f'Pathway from {self.source} to {self.target}:')
        for j in range(len(self.idx_smi)):
            path = self.pred_paths[self.idx_smi[j]]
            print(path)

            if with_reactions:
                reactions = self.get_pathway_reactions(path)
                print(reactions)

            if to_bigg:
                bigg_pathway = self.kegg_to_bigg_pathway(path)
                print(bigg_pathway)

                if with_reactions:
                    bigg_reactions = []
                    for rxn_list in reactions:
                        bigg_reactions.append([self.kegg_to_bigg_reaction(rxn) for rxn in rxn_list])
                    print(bigg_reactions)

                print('\n')

    def kegg_to_bigg_pathway(self, kegg_pathway: list) -> list:
        bigg_pathway = [self.kegg_to_bigg_compound(cpd) for cpd in kegg_pathway]
        return bigg_pathway
    
    ''' Function to define self.path_compound and self.path_reactions given a single pathway '''
    def single_pathway(self, path: list):
        self.path_compound = path
        self.path_reactions = self.get_pathway_reactions(path)

    ''' list of arrays to list of lists '''
    def _list_of_arrays_to_list_of_lists(self, l: list) -> list:
        return [list(x) for x in l]

    ''' get reactions between pathway compounds '''
    def get_pathway_reactions(self, path: list, to_bigg=False) -> list:
        reactions = []
        for i in range(len(path)-1):
            cpd_a = path[i]
            cpd_b = path[i+1]
            reactions.append(self.graph.get_reaction_by_compounds(cpd_a, cpd_b))

        reactions = self._list_of_arrays_to_list_of_lists(reactions)
        return reactions            

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
       
    def __str__(self):
        return f'Pathway from {self.source} to {self.target}'
