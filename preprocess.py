import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, RDConfig, rdmolfiles, Draw, AllChem
import statistics
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
import selfies as sf
from Vocabulary import Vocabulary

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Calculate Lipinski Descriptors
def lipinski(smiles, verbose=False):
    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

# IC50 to pIC50 for better distribution
def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x

# Normalization for conversion
def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if float(i) > 100000000:
          i = 100000000
        norm.append(float(i))

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)
        
    return x

def preview(search, id):
    # Get Info From Database
    target = new_client.target
    target_query = target.search(search)
    targets = pd.DataFrame.from_dict(target_query)
    print(targets.target_chembl_id)

# Get all information of inhibitors for target protein from Chembl
def preprocess(search, chembl_id, acronym):
    # Get Info From Database
    '''target = new_client.target
    target_query = target.search(search)
    targets = pd.DataFrame.from_dict(target_query)
    index = list(targets.target_chembl_id).index(chembl_id)
    selected_target = targets.target_chembl_id[index]
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    df.to_csv(acronym + '_bioactivity_data.csv', index=False)'''

    # Get Important Vals
    '''df2 = df[df.standard_value.notna()]
    mol_cid = []
    for i in df2.molecule_chembl_id:
        mol_cid.append(i)
    canonical_smiles = []
    for i in df2.canonical_smiles:
        canonical_smiles.append(i)
    standard_value = []
    for i in df2.standard_value:
        standard_value.append(i)
    data_tuples = list(zip(mol_cid, canonical_smiles, standard_value))
    df3 = pd.DataFrame(data_tuples,  columns=['chembl_id', 'canonical_smiles', 'standard_value'])
    df3.to_csv(acronym + '_bioactivity_preprocessed_data.csv', index=False)'''

    df = pd.read_csv('cetp_bioactivity_data.csv')
    df3 = pd.read_csv('cetp_bioactivity_preprocessed_data.csv')

    # Remove duplicate canonical SMILES
    # Average standard values
    df3 = remove_duplicate_SMILES_w_standard_value(df3)
    df3.to_csv(acronym + '_bioactivity_filtered.csv', index=False)

    # Lipinski Stuff
    df_lipinski = lipinski(df3.canonical_smiles)
    df_combined = pd.concat([df3,df_lipinski], axis=1)

    df_norm = norm_value(df_combined)
    df_final = pIC50(df_norm)

    # QED
    smiles = list(df_final['canonical_smiles'])
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    qed = [Chem.QED.default(m) for m in mols]
    df_final['QED'] = qed

    # SAS
    sas = [sascorer.calculateScore(m) for m in mols]
    df_final['SAS'] = sas

    df_final.to_csv(acronym + '_master_data.csv', index=False)

def get_canonical_SMILES():
    chembl = open('Small Molecules/chembl_33_chemreps.txt', 'r')

    # Burn first line (header)
    chembl.readline()

    chembl_id = []
    canonical_smiles = []
    cur_line = chembl.readline()
    while cur_line != '':
        cur_vals = cur_line.split('\t')
        chembl_id.append(cur_vals[0])
        canonical_smiles.append(cur_vals[1])
        cur_line = chembl.readline()

    df = pd.DataFrame()
    df['chembl_id'] = chembl_id
    df['canonical_smiles'] = canonical_smiles
    print(len(df))
    df.to_csv('Small Molecules/chembl_database.csv')

# Remove duplicates based on SMILES
def remove_duplicate_SMILES(df, path):
    ids = list(df['chembl_id'])
    smiles = list(df['canonical_smiles'])
    filtered_ids = []
    filtered_smiles = []
    prev_smiles = []
    for i, sm in enumerate(smiles):
        if sm not in prev_smiles:
            filtered_ids.append(ids[i])
            filtered_smiles.append(sm)
            prev_smiles.append(sm)

    print(str(len(filtered_smiles)))

    new_df = pd.DataFrame()
    new_df['chembl_id'] = filtered_ids
    new_df['canonical_smiles'] = filtered_smiles
    new_df.to_csv(path + 'filtered_chembl_database.csv')

# Remove duplicate SMILES and average their pIC50's
def remove_duplicate_SMILES_w_standard_value(df):
    all_smiles = list(df['canonical_smiles'])
    all_ids = list(df['chembl_id'])
    all_values = list(df['standard_value'])
    existing_smiles = []
    repeating_smiles = []

    filtered_id = []
    filtered_sv = []
    for i, sm in enumerate(all_smiles):
        if sm in existing_smiles:
            if sm not in repeating_smiles:
                repeating_smiles.append(sm)
        else:
            existing_smiles.append(sm)
            filtered_id.append(all_ids[i])
            filtered_sv.append(all_values[i])

    for sm in repeating_smiles:
        # Get indices of all occurences of current repeating SMILES
        indices = [i for (i, s) in enumerate(all_smiles) if s==sm]

        # Get all standard values for this repeating smiles
        all = [all_values[i] for i in indices]
        print(all)
        avg = statistics.mean(all)
        filtered_sv[existing_smiles.index(sm)] = avg

    # Save to df
    new_df = pd.DataFrame()
    new_df['chembl_id'] = filtered_id
    new_df['canonical_smiles'] = existing_smiles
    new_df['standard_value'] = filtered_sv

    return new_df

# Sample SMILES from Chembl dataset
def sample_SMILES(num, length_threshold=100):
    df = pd.read_csv('Small Molecules/Datasets/filtered_chembl_database.csv')
    df = df.sample(frac=1, ignore_index=True)

    subset_id = []
    subset_smiles = []
    index = 0
    success = 0
    while success < num:
        cur_smiles = df.at[index, 'canonical_smiles']
        if len(cur_smiles) < length_threshold:
            subset_id.append(df.at[index, 'chembl_id'])
            subset_smiles.append(cur_smiles)
            success += 1
        index += 1
    new_df = pd.DataFrame()
    new_df['chembl_id'] = subset_id
    new_df['canonical_smiles'] = subset_smiles
    new_df.to_csv('Small Molecules/' + str(num) + '_chembl_subset.csv')

# Remove all SMILES with length less than threshold
def SMILES_cutoff(length_threshold=100):
    df = pd.read_csv('Small Molecules/BACE1/bace1_master_data.csv')

    smiles = list(df['canonical_smiles'])
    indices = []
    for i in range(len(smiles)):
        # CHANGE TO <=
        if len(smiles[i]) < length_threshold:
            indices.append(i)
    
    new_df = df.loc[indices]
    new_df = new_df.reset_index()
    new_df.drop(labels=['index','Unnamed: 0'] , axis=1, inplace=True)
    new_df.to_csv('Small Molecules/BACE1/BACE1.csv')

# Find the molecular properties for given SMILES
def find_molecular_properties(df, path, properties_to_add=['Lipinski', 'QED', 'SAS'], save=True):
    smiles = list(df['canonical_smiles'])
    mols = [Chem.MolFromSmiles(str(sm)) for sm in smiles]

    if None in mols:
        print("FAIL")
        print(mols.index(None))

    if 'Lipinski' in properties_to_add:
        df_lipinski = lipinski(smiles)
        df = pd.concat([df, df_lipinski], axis=1)
        #df.drop(labels='Unnamed: 0', axis=1, inplace=True)

    if 'QED' in properties_to_add:
        qed = [Chem.QED.default(m) for m in mols]
        df['QED'] = qed

    if 'SAS' in properties_to_add:
        sas = []
        for m in mols:
            try:
                sas.append(sascorer.calculateScore(m))
            except ZeroDivisionError:
                # If SAS cannot be calculated, make it as high as possible (unsynthesizable)
                sas.append(10)
        df['SAS'] = sas

    if save:
        df.to_csv(path, index=False)
    return df

# Remove invalid SMILES
def remove_invalid_SMILES(df, path):
    for i in range(len(df)):
        sm = df.at[i, 'canonical_smiles']
        if Chem.MolFromSmiles(sm) == None:
            df.drop(labels=i, inplace=True)
    df.reset_index(inplace=True)
    df.to_csv(path + 'filtered2_chembl_database.csv')

# Convert SMILES to SELFIES
# Remove SMILES that don't follow default SELFIES constraints
# Remove SMILES that aren't properly converted from SMILES --> SELFIES --> SMILES
def SMILES_to_SELFIES(df, save_path):
    columns = list(df.columns)
    all_selfies = []
    counter = 0
    for i in range(len(df)):
        failed = False
        try:
            sm = df.at[i, 'canonical_smiles']
            selfies = sf.encoder(sm)
            dec_sm = sf.decoder(selfies)
            dec_sm = Chem.CanonSmiles(dec_sm)
            sm = Chem.CanonSmiles(sm)
            if dec_sm != sm:
                print("Error at index " + str(i) + ": Failed Match")
                failed = True
                counter += 1
            else:
                all_selfies.append(selfies)
        except sf.exceptions.EncoderError:
            print("Error at index " + str(i) + ": Failed Conversion")
            failed = True
            counter += 1
        if failed:
            df.drop(labels=i, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['selfies'] = all_selfies
    columns.insert(2, 'selfies')
    df = df[columns]
    df.to_csv(save_path, index=False)
    print('Total Fails: ' + str(counter))

def sample_SELFIES(df, num, length_threshold):
    # Randomly shuffle dataframe
    df = df.sample(frac=1, ignore_index=True)

    indices = []
    index = 0
    success = 0
    while success < num:
        cur_selfies = df.at[index, 'selfies']
        if sf.len_selfies(cur_selfies) < length_threshold:
            indices.append(index)
            success += 1
        index += 1
    
    new_df = df.loc[indices]
    new_df.reset_index(inplace=True, drop=True)
    return new_df

def create_datasets(df_chembl, df_target, quantities, save_path, length_threshold=100, vocab_datasets = [250000, 500000]):
    target_vocab = Vocabulary(list(df_target['selfies']))

    for n in quantities:
        success = False
        while not success:
            df = sample_SELFIES(df_chembl, n, length_threshold)
            vocab = Vocabulary(list(df['selfies']))
            if set(target_vocab.unique_chars).issubset(set(vocab.unique_chars)) or n not in vocab_datasets:
                success = True
                name = str(int(n / 1000)) + 'k'
                df.to_csv(save_path + name + '_subset.csv')
                print(str(n) + ' Completed')
            else:
                print(str(n) + ' Failed')

def SELFIES_cutoff(df, length_threshold=100):
    indices = []
    for i in range(len(df)):
        if sf.len_selfies(df.at[i, 'selfies']) < length_threshold:
            indices.append(i)
    
    new_df = df.loc[indices]
    new_df.reset_index(inplace=True, drop=True)
    new_df.to_csv('Small Molecules/BACE1/BACE1.csv')

def remove_period_SMILES(df, save_path):
    for i in range(len(df)):
        if '.' in df.at[i, 'canonical_smiles']:
            df.drop(labels=i, inplace=True)
        if (i % 1000) == 0:
            print(i)
    df.reset_index(inplace=True, drop=True)
    df.to_csv(save_path, index=False)

def sample_subset_selfies(vocab_df, num, length_threshold):
    dataset_df = pd.read_csv('SeawulfGAN/Data/500k_subset.csv')
    all_vocabs = Vocabulary(list(vocab_df['selfies']))
    
    success = False
    while not success:
        gen_df = sample_SELFIES(dataset_df, num, length_threshold)
        vocab = Vocabulary(list(gen_df['selfies']))
        success = set(vocab.unique_chars).issubset(set(all_vocabs.unique_chars))
        print(success)
    #gen_df.drop(labels='Unnamed: 0', axis=1, inplace=True)
    gen_df.to_csv('SeawulfGAN/Data/100k_subset_500k.csv', index=False)

# Double checks that all smiles are canonical
def make_canonical(df, save_path):
    smiles = list(df['canonical_smiles'])
    can_smiles = [Chem.CanonSmiles(sm) for sm in smiles]
    df['canonical_smiles'] = can_smiles
    df.to_csv(save_path, index=False)