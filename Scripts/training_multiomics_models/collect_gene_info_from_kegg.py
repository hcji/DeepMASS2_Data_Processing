# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:14:06 2023

@author: DELL
"""


import re
import requests
import numpy as np
import pandas as pd
 
from tqdm import tqdm
from bs4 import BeautifulSoup


output = {}
kegg_compounds = pd.read_csv('D:/All_Database/KEGG/kegg_compounds.csv')
for k in tqdm(kegg_compounds['ID'][16796:]):
    for i in range(100):
        try:
            url = 'https://www.genome.jp/entry/{}'.format(k)
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.find('table', class_='w2')
            break
        except ConnectionError:
            pass
    if table is None:
        continue
    for row in table.find_all('tr'):
        cells = row.find_all(['th', 'td'])
        if len(cells) == 2:
            key = cells[0].text.strip()
            if key == 'Gene':
                value = cells[1].text.strip()
                genes = value.split('; ')
                output[k] = genes

np.save('output.npy', output)

