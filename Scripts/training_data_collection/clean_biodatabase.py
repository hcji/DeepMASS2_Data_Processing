# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:20:34 2023

@author: DELL
"""

import requests
import pandas as pd
from tqdm import tqdm

biodatabase = pd.read_csv('Saves/public_version/DeepMassStructureDB-v1.0.csv')

def get_synonyms_by_inchikey(inchikey):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    operation = "inchikey/" + inchikey + "/synonyms/JSON"
    url = f"{base_url}/{operation}"
    response = requests.get(url)
    output = []
    if response.status_code == 200:
        data = response.json()
        synonyms = data["InformationList"]["Information"]
        for synonyms_i in synonyms:
            try:
                output = synonyms_i["Synonym"]
                continue
            except:
                pass
        return output
    else:
        print(f"Error: Unable to retrieve synonyms. Status code: {response.status_code}")
        return []


synonyms = []
for i in tqdm(biodatabase.index): 
    inchikey = biodatabase['InChIkey'][i]
    synonyms_i = get_synonyms_by_inchikey(inchikey)
    if len(synonyms_i) == 0:
        synonyms_i = biodatabase['Title'][i]
    else:
        synonyms_i = synonyms_i[0]
    synonyms.append(synonyms_i)
        


