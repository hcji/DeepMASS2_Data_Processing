# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:39:23 2023

@author: DELL
"""


import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm


def read_hmdb_fasta_file_as_dictionary(file_name):
    fasta_file = open(file_name, 'r')
    main_dictionary_of_sequences = {}
    last_id = ""
    for line in fasta_file:
        if line[:4] == 'HMDB':
            blank_string = ""
            main_dictionary_of_sequences.update({line[:10]: blank_string})
            last_id = line[:10]
        else:
            last_line = main_dictionary_of_sequences[last_id]
            updated_line = last_line + line.replace('\n', '')
            main_dictionary_of_sequences.update({last_id: updated_line})
    fasta_file.close()
    return main_dictionary_of_sequences

hmdb_protein = read_hmdb_fasta_file_as_dictionary('Datasets/HMDB/protein.fasta')


xml_file = "E:/Data/hmdb_metabolites/hmdb_metabolites.xml"
tree = ET.parse(xml_file)
root = tree.getroot()
metabolites = []

for metabolite_elem in tqdm(root.findall('{http://www.hmdb.ca}metabolite')):
    metabolite = {}

    # Extract information for each metabolite
    metabolite['accession'] = metabolite_elem.find('{http://www.hmdb.ca}accession').text
    metabolite['name'] = metabolite_elem.find('{http://www.hmdb.ca}name').text
    metabolite['smiles'] = metabolite_elem.find('{http://www.hmdb.ca}smiles').text
    
    protein_associations = []
    protein_associations_tree = metabolite_elem.find('{http://www.hmdb.ca}protein_associations')
    if protein_associations_tree is not None:
        for protein_elem in protein_associations_tree.findall('{http://www.hmdb.ca}protein'):
            protein = {}
            protein['protein_accession'] = protein_elem.find('{http://www.hmdb.ca}protein_accession').text
            protein['protein_name'] = protein_elem.find('{http://www.hmdb.ca}name').text
            protein['uniprot_id'] = protein_elem.find('{http://www.hmdb.ca}uniprot_id').text
            protein['gene_name'] = protein_elem.find('{http://www.hmdb.ca}gene_name').text
            protein['protein_type'] = protein_elem.find('{http://www.hmdb.ca}protein_type').text
            protein_associations.append(protein)
            metabolite['protein_associations'] = protein_associations
    if len(protein_associations) > 0:
        metabolites.append(metabolite)


hmdb_pid, hmdb_hid, hmdb_seq, prot_smiles = [], [], [], []
for metabolite_elem in metabolites:
    prot_smiles_i = metabolite_elem['smiles']
    for prot in metabolite_elem['protein_associations']:
        hmdb_pid_i = prot['uniprot_id']
        hmdb_hid_i = prot['protein_accession']
    if hmdb_pid_i not in hmdb_pid:
        try:
            hmdb_seq_i = hmdb_protein[hmdb_hid_i]
        except:
            hmdb_seq_i = ''
        hmdb_pid.append(hmdb_pid_i)
        hmdb_hid.append(hmdb_hid_i)
        hmdb_seq.append(hmdb_seq_i)
        prot_smiles.append([prot_smiles_i])
    else:
        j = hmdb_pid.index(hmdb_pid_i)
        prot_smiles[j] += [prot_smiles_i]

hmdb_protein = pd.DataFrame({'Uniprot ID': hmdb_pid,
                             'HMDB ID': hmdb_hid,
                             'Sequence': hmdb_seq,
                             'Association SMILES': prot_smiles})
np.save('Datasets/HMDB/metabolite_protein_associations.npy', hmdb_protein)

