import pandas as pd
import numpy as np
from tqdm import tqdm
from data import Data
from compound import Compound
from reaction import Reaction

class Microorganism:
    def __init__(self, name: str):
        self.name = name
        