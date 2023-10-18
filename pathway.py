# Add path to the model
import sys
sys.path.append("../")
from microorganism import Microorganism
from data import Data
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
        # Store the selected pathway reactions after the def select_reactions()
        self.selected_pathway_reactions = None
        # Store directions of each selected reaction
        self.direction = None
        # Store metabolites available in the host microorganism
        self.cobra_kegg_mets = None

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
    
    ''' Update self.cobra_kegg_mets with metabolites available in the host microorganism '''
    def _update_cobra_kegg_mets(self, cobra_model: Microorganism) -> list:
        self.cobra_kegg_mets = cobra_model.metabolites_df['kegg'].tolist()

    ''' Function to define self.path_compound and self.path_reactions given a single pathway '''
    def single_pathway(self, path: list):
        self.path_compound = path
        self.path_reactions = self.get_pathway_reactions(path)
        self.selected_pathway_reactions = []
        self.direction = []
        return self.path_compound, self.path_reactions

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
        product_list = []
        
    ''' Select reactions for a pathway when multiple reactions for each step are proposed '''
    def select_reactions(self, data: Data, cobra_model: Microorganism, verbose=False) -> list:
        self.selected_pathway_reactions = []

        # Get available metabolites in the host microorganism
        self._update_cobra_kegg_mets(cobra_model)

        product_list = []
        j = 0
        for r in self.path_reactions:
            for r_i in r:
                # Check the part of reeaction that has the src metabolite and set it as the reactants part
                if self.path_compound[j] in data.reactions[r_i].compounds[0]:
                    products = data.reactions[r_i].compounds[1]
                    reactants = data.reactions[r_i].compounds[0]
                elif self.path_compound[j] in data.reactions[r_i].compounds[1]:
                    products = data.reactions[r_i].compounds[0]
                    reactants = data.reactions[r_i].compounds[1]
                else:
                    return 'Error: metabolite not found in reaction'
                
                # Update products list
                for p_i in products:
                    product_list.append(p_i)
                # keep only unique elements in the product_list
                product_list = list(set(product_list))

                if verbose:
                    print(f'Reaction {r_i}, {data.reactions[r_i].equation}')
                    print(f'Pathway compound: {self.path_compound[j]}')
                    print(f'Products: {products}')
                    print(f'Reactants: {reactants}')
                    print(f'Product list: {product_list}')

                # Check if the reactans are in the cobra_kegg_mets or in product_list
                is_available = False
                for r_j in reactants:
                    if r_j in self.cobra_kegg_mets or r_j in product_list:
                        is_available = True
                        continue
                    else:
                        is_available = False
                        break

                if is_available:
                    self.selected_pathway_reactions.append(r_i)
                    break
                else:
                    break

                print()
            
            j += 1

            if verbose:
                print(f"The selected pathway reactions are: {self.selected_pathway_reactions}")        
        
        if(len(self.selected_pathway_reactions) == len(self.path_reactions)):
            pass
        else:
            raise ValueError('Error! Did not selected reactions')
        
    def _str_map(self, string, df: pd.DataFrame, arrow: str, col = 'metabolites') -> str:
        # Replace <=> with --> in string
        string = string.replace('<=>', arrow)
        for i in range(len(df)):
            string = string.replace(df.iloc[i]['kegg'], df.iloc[i][col])
        return string
    
    ''' Save reactions to add to GEM to file '''
    def _reactions_to_file(self, EQS: list, filename: str) -> None:
        df = pd.DataFrame()
        df['Reaction'] = ['R' + str(i) for i in range(1, len(EQS)+1)]
        df['Equation'] = EQS
        df.to_csv(filename, index=False)

    ''' Prints the reactions to add to GEM in wanted form'''
    def reactions_add_gem(self, data: Data, cobra_model: Microorganism, col = 'metabolites', save=False) -> list:
        self._get_reactions_direction(data, cobra_model)
        EQS = []
        for i, rxn in enumerate(self.selected_pathway_reactions):
            equation = data.reactions[rxn].equation
            arrow = self.direction[i]
            equation = self._str_map(equation, cobra_model.metabolites_df, arrow, col)
            EQS.append(equation)
        
        if save:
            filename = f'../results/{self.source}_{self.target}_{col}_reactions.csv'
            self._reactions_to_file(EQS, filename)
        return EQS

    ''' Updates self.direction with the correct direction for every reaction to be added to microorganism'''
    def _get_reactions_direction(self, data: Data, cobra_model: Microorganism) -> None:
        available_metabolites = cobra_model.metabolites_df['kegg'].values.tolist()
        for r in self.selected_pathway_reactions:
            reactants = data.reactions[r].compounds[0]
            products = data.reactions[r].compounds[1]

            right_arrow = (all([p in available_metabolites for p in reactants]))
            left_arrow = (all([p in available_metabolites for p in products]))

            if right_arrow: self.direction.append('-->')
            elif left_arrow: self.direction.append('<--')
            else: 
                raise ValueError('Error! Not able to set reaction direction')
                
            if right_arrow:
                for p in products:
                    available_metabolites.append(p)
            elif left_arrow:
                for p in reactants:
                    available_metabolites.append(p)
            else:
                raise ValueError('Error! Not able to set reaction direction')
                
            
    def __str__(self):
        return f'Pathway from {self.source} to {self.target}'
