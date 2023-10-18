import pandas as pd
import numpy as np
from tqdm import tqdm
from data import Data
from compound import Compound
from reaction import Reaction
import cobra

class Microorganism:
    def __init__(self, cobra_model):
        self.cobra_model = cobra_model
        self.metabolites_names = [m.name for m in self.cobra_model.metabolites]
        self.metabolites_id = [m.id for m in self.cobra_model.metabolites]
        self.reactions_names = [r.name for r in self.cobra_model.reactions]
        self.reactions_id = [r.id for r in self.cobra_model.reactions]
        self.name = self.cobra_model._id
        self.metabolites_df = None
        self.reactions_df = None

        self._create_metabolites_df()
        self._create_reactions_df()

    def _create_metabolites_df(self):
        metabolites_kegg = []
        metabolites_seed = []
        compartments = []
        for m in self.metabolites_id:
            try:
                kegg_id = self.cobra_model.metabolites.get_by_id(m)._annotation['kegg.compound']
                metabolites_kegg.append(kegg_id[0])
            except KeyError:
                kegg_id = "NA"
                metabolites_kegg.append(kegg_id)
            try:
                seed_id = self.cobra_model.metabolites.get_by_id(m)._annotation['seed.compound']
                metabolites_seed.append(seed_id[0])
            except KeyError:
                seed_id = "NA"
                metabolites_seed.append(seed_id)  
            comp = self.cobra_model.metabolites.get_by_id(m).compartment
            compartments.append(comp)

        # Create dataframe with metabolites and their KEGG ID
        self.metabolites_df = pd.DataFrame({"metabolites": self.metabolites_id, 
                                            "kegg": metabolites_kegg, 
                                            "seed": metabolites_seed,
                                            "compartment": compartments})

    def _create_reactions_df(self):
        reactions_kegg = []
        compartments = []
        for r in self.reactions_id:
            try:
                kegg_id = self.cobra_model.reactions.get_by_id(r)._annotation['kegg.reaction']
                reactions_kegg.append(kegg_id[0])
            except KeyError:
                kegg_id = 'NA'
                reactions_kegg.append(kegg_id)

        # Create dataframe with metabolites and their KEGG ID
        self.reactions_df = pd.DataFrame({"reactions": self.reactions_id, "kegg": reactions_kegg})


    def __str__(self) -> str:
        return f"{self.cobra_model._id} ({len(self.metabolites_id)} metabolites, {len(self.reactions_id)} reactions)"
    
    