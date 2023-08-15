# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:20:34 2023

@author: DELL
"""

import requests
import pandas as pd
from tqdm import tqdm

biodatabase = pd.read_csv('Saves/public_version/DeepMassStructureDB-v1.0.csv')


def capitalize_first_letter(input_string):
    result = ""
    capitalize_next = True

    for char in input_string:
        if char.isalpha():
            if capitalize_next:
                result += char.upper()
                capitalize_next = False
            else:
                result += char.lower()
        else:
            result += char
            capitalize_next = True

    return result


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
    synonyms_i = str(biodatabase['Title'][i])
    if (synonyms_i[:4] == 'UNPD') or (synonyms_i[:3] == 'CNP') or (synonyms_i == 'nan') or (synonyms_i == ''):
        inchikey = biodatabase['InChIkey'][i]
        try:
            synonyms_i = get_synonyms_by_inchikey(inchikey)
            if len(synonyms_i) > 0:
                synonyms_i = synonyms_i[0]
        except:
            pass
    if synonyms_i == '':
        synonyms_i = biodatabase['Title'][i]
    synonyms_i = capitalize_first_letter(synonyms_i)
    synonyms.append(synonyms_i)

biodatabase['Title'] = synonyms
biodatabase.to_csv('Saves/public_version/DeepMassStructureDB-v1.0.csv')
