import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, RDConfig, Draw, AllChem, rdmolfiles
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
import statistics
import math
import selfies as sf
import visualize
import os
from AE import Autoencoder as AE
from Vocabulary import Vocabulary
from Vocabulary2 import Vocabulary as Vocabulary2
import preprocess
from predictor_lv import Predictor
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
import random

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Enumerate SMILES
# Takes canonical SMILES as input
def randomize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)

# METHODS TO EXAMINE DUPLICATES IN DATA
def check_duplicates(dfs, bioactivity='pIC50'):
    for df in dfs:
        check_duplicate_SMILES(df)
        check_duplicate_ID(df)
        if bioactivity in df.columns:
            check_duplicate_value(df, bioactivity)
        print('\n')

def check_duplicate_SMILES(df):
    all_smiles = list(df['canonical_smiles'])
    total_len = len(all_smiles)
    filtered_smiles = set(all_smiles)
    filtered_len = len(filtered_smiles)

    n_duplicates = total_len - filtered_len
    print(n_duplicates)
    return n_duplicates, total_len, filtered_len

def check_duplicate_ID(df):
    all_id = list(df['chembl_id'])
    total_len = len(all_id)
    filtered_id = set(all_id)
    filtered_len = len(filtered_id)

    n_duplicates = total_len - filtered_len
    print(n_duplicates)
    return n_duplicates, total_len, filtered_len

def check_duplicate_bioactivity(df, bioactivity):
    all_ba = list(df[bioactivity])
    total_len = len(all_ba)
    filtered_ba = set(all_ba)
    filtered_len = len(filtered_ba)

    n_duplicates = total_len - filtered_len
    print(n_duplicates)
    return n_duplicates, total_len, filtered_len

def check_duplicate_value(df, value):
    all_smiles = list(df['canonical_smiles'])
    all_values = list(df[value])
    existing_smiles = []
    repeating_smiles = []

    repeated_smiles = 0
    for sm in all_smiles:
        if sm in existing_smiles:
            repeated_smiles += 1
            if sm not in repeating_smiles:
                repeating_smiles.append(sm)
        else:
            existing_smiles.append(sm)
    
    repeated_value = 0
    all_differences = []
    for sm in repeating_smiles:
        # Get indices of all occurences of current repeating SMILES
        indices = [i for (i, s) in enumerate(all_smiles) if s==sm]

        # Get number of repeated values amongst the repeating SMILES
        all = []
        for i in indices:
            if all_values[i] in all:
                repeated_value += 1
            else:
                all.append(all_values[i])

        # Calculate average pairwise difference
        all = [all_values[i] for i in indices]
        differences = pairwise_difference(all)
        all_differences.extend(differences)
    
    print('Repeating SMILES: ' + str(repeated_smiles))
    print('Repeating ' + value + ': ' + str(repeated_value))
    print('Average Difference: ' + str(round(statistics.mean(all_differences), 5)))
    print('Average Stdev: ' + str(round(statistics.stdev(all_differences), 5)))
    return repeated_smiles, repeated_value

def remove_duplicates(df, save_path):
    check_duplicate_SMILES(df)

    total_length = len(df)
    df.drop_duplicates(subset='canonical_smiles', inplace=True)
    print(len(df) - total_length)
    print(len(df) / total_length)
    df.to_csv(save_path, index=False)

def pairwise_difference(vals):
    differences = []
    for i in range(len(vals)-1):
        for j in range(len(vals)-i-1):
            differences.append(abs(vals[i] - vals[i+j+1]))
    return differences

# Make sure that BACE vocab is within AE vocab
def check_vocab(df_target, df_ae):
    smiles_target = list(df_target['selfies'])
    smiles_ae = list(df_ae['selfies'])

    vocab_target = Vocabulary(smiles_target)
    vocab_ae = Vocabulary(smiles_ae)

    return set(vocab_target.unique_chars).issubset(set(vocab_ae.unique_chars))

# Compare string length and tokenized length b/w SMILES and SELFIES
def SMILES_to_SELFIES_length_comparison(df):
    length_ratio = []
    token_ratio = []
    smiles_lengths = []
    token_lengths = []
    for i in range(len(df)):
        sm = df.at[i, 'canonical_smiles']
        se = df.at[i, 'selfies']
        length_ratio.append(len(se) / len(sm))
        token_ratio.append(sf.len_selfies(se) / len(sm))
        smiles_lengths.append(len(sm))
        token_lengths.append(sf.len_selfies(se))
    print('Length Ratio: ' + str(statistics.mean(length_ratio)))
    print('Token Ratio: ' + str(statistics.mean(token_ratio)))
    print('Token Ratio Stdev: ' + str(statistics.stdev(token_ratio)))
    print('Avg SMILES Length: ' + str(statistics.mean(smiles_lengths)))
    print('Avg SELFIES Tokenized Length: ' + str(statistics.mean(token_lengths)))

# Check SMILES --> SELFIES --> SMILES
def SELFIES_to_SMILES_check(df):
    fail_counter = 0
    for i in range(len(df)):
        dec_sm = sf.decoder(df.at[i, 'selfies'])
        dec_sm = Chem.CanonSmiles(dec_sm)
        sm = Chem.CanonSmiles(df.at[i, 'canonical_smiles'])
        if dec_sm != sm:
            print("FAIL")
            fail_counter += 1
    print(fail_counter)

def SELFIES_parse_check(df):
    fail_counter = 0
    for i in range(len(df)):
        try:
            parsed = sf.split_selfies(df.at[i, 'selfies'])
        except ValueError:
            print("Fail Index " + str(i))
            fail_counter += 1
    print(fail_counter)

def check_token_lengths(folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if 'subset' in f]
    for f in files:
        df = pd.read_csv(folder_path + f)
        selfies = list(df['selfies'])
        print(f + ': ' + str(len(sf.get_alphabet_from_selfies(selfies)) + 3))

def calculate_qed(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    qeds = [Chem.QED.default(m) for m in mols if m != None]
    return qeds

def calculate_sas(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]

    sas = []
    for m in mols:
        try:
            sas.append(sascorer.calculateScore(m))
        except ZeroDivisionError:
            continue
    
    return sas

def calculate_MW(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    mws = [Descriptors.MolWt(m) for m in mols if m != None]
    return mws

def calculate_LogP(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    logps = [Descriptors.MolLogP(m) for m in mols if m != None]
    return logps

def generation_distributions(train_df, train_name, included_epochs, folder_path, metrics = ['QED', 'SAS', 'pIC50', 'MW', 'LogP']):
    train_vals = [list(train_df[metric]) for metric in metrics]

    all_files = os.listdir(folder_path)
    all_files = [f for f in all_files if 'samples_epoch_' in f]
    epochs = [int(f.split('samples_epoch_')[1].split('.txt')[0]) for f in all_files]
    all_files = [val for (_, val) in sorted(zip(epochs, all_files), key=lambda x:x[0])]
    epochs.sort()

    # Load predictor model
    main_path = prefix + 'SeawulfGAN/Models/' 
    vocab_path = prefix + 'SeawulfGAN/Data/500k_subset.csv'
    ae_path = prefix + 'SeawulfGAN/Models/AE_model_500k.h5'
    predictor_path = prefix + 'SeawulfGAN/Models/'
    predictor_dataset = prefix + 'SeawulfGAN/Data/BACE1.csv'
    suffix = '_500k'
    _, _, predictor = initialize_models(main_path, vocab_path, ae_path, predictor_path, predictor_dataset, suffix)

    all_qed = []
    all_sas = []
    all_pIC50 = []
    all_mw = []
    all_logp = []
    all_epoch_names = []
    for i in range(len(all_files)):
        if epochs[i] in included_epochs:
            f = open(folder_path+all_files[i], 'r')
            f.readline()
            all_selfies = f.read()
            all_selfies = all_selfies.split('\n')

            if 'QED' in metrics:
                all_qed.append(calculate_qed(all_selfies))
            if 'SAS' in metrics:
                all_sas.append(calculate_sas(all_selfies))
            if 'pIC50' in metrics:
                p_vals = predictor.predict(all_selfies)
                all_pIC50.append([v for v in p_vals if v > -10 and v < 20])
            if 'MW' in metrics:
                all_mw.append(calculate_MW(all_selfies))
            if 'LogP' in metrics:
                all_logp.append(calculate_LogP(all_selfies))
            all_epoch_names.append('Epoch ' + str(epochs[i]))
    
    gen_vals = []
    if 'QED' in metrics:
        gen_vals.append(all_qed)
    if 'SAS' in metrics:
        gen_vals.append(all_sas)
    if 'pIC50' in metrics:
        gen_vals.append(all_pIC50)
    if 'MW' in metrics:
        gen_vals.append(all_mw)
    if 'LogP' in metrics:
        gen_vals.append(all_logp)

    names = [train_name]
    names.extend(all_epoch_names)
    for i in range(len(metrics)):
        datapoints = [train_vals[i]]
        datapoints.extend(gen_vals[i])
        visualize.compare_property_distribution2(datapoints, names, metrics[i], 'Generated Distributions', folder_path + 'distribution_' + metrics[i] + '.png')

def generation_distributions_single(train_df, train_name, folder_path):
    train_qed = list(train_df['QED'])
    all_files = os.listdir(folder_path)
    all_files = [f for f in all_files if 'samples_epoch_' in f]
    epochs = [f.split('samples_epoch_')[1].split('.txt')[0] for f in all_files]

    for i in range(len(all_files)):
        f = open(folder_path+all_files[i], 'r')
        f.readline()
        all_selfies = f.read()
        all_selfies = all_selfies.split('\n')
        gen_qed = calculate_qed(all_selfies)
        visualize.compare_property_distribution2([train_qed, gen_qed], [train_name, 'Epoch '+epochs[i]], 'QED', 'Generated Distribution Epoch ' + epochs[i], folder_path + 'distribution_epoch_' + epochs[i] + '.png')

def period_char_analysis(print_mols=False, folder_path=None):
    df = pd.read_csv('Small Molecules/Datasets/filtered_chembl_selfies2.csv')
    
    counter = 0
    fail_counter = 0
    for i in range(len(df)):
        if '.' in df.at[i, 'canonical_smiles']:
            in_smiles = True
        else:
            in_smiles = False
        if '.' in df.at[i, 'selfies']:
            in_selfies = True
        else:
            in_selfies = False
        
        if in_smiles or in_selfies:
            counter += 1
            if print_mols:
                m = Chem.MolFromSmiles(df.at[i, 'canonical_smiles'])
                Draw.MolToFile(m, folder_path + 'sample_' + str(i) + '.png')

        if in_smiles != in_selfies:
            print("FAIL")
            fail_counter += 1
    
    print(fail_counter)
    print(counter)

# Returns how similar or diverse molecules are relative to each other
def internal_diversity(smiles):
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    #fps = [AllChem.GetMorganFingerprintAsBitVect(m,radius=2, nBits=1024) for m in mols if m != None]
    fps = [GetMACCSKeysFingerprint(m) for m in mols if m != None]

    all_similarities = []
    for i in range(len(fps)-1):
        all_similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))

    if len(all_similarities) == 0:
        return 0
    else:
        return 1 - statistics.mean(all_similarities)

# Determines percentage of generated selfies that are exact duplicates
# Another measure of similarity/diversity
def duplicates(selfies):
    total_length = len(selfies)
    indiv_length = len(set(selfies))
    return indiv_length / total_length

def autoencoder_diversity(n_samples, vocab_path, ae_path):
    main_path = 'AE Diversity/'
    vocab, auto, _ = initialize_models(main_path, vocab_path, ae_path)
    
    df = pd.read_csv(vocab_path)
    selfies = list(df['selfies'])

    # Get latent vectors of training data
    tok = vocab.tokenize(selfies)
    encoded = vocab.encode(tok)
    latent_vectors = auto.sm_to_lat_model.predict(encoded)

    # Reshape latent vectors based on each index
    reshaped_lvs = []
    for j in range(len(latent_vectors[0])):
        cur = []
        for i in range(len(latent_vectors)):
            cur.append(latent_vectors[i][j])
        reshaped_lvs.append(cur)

    # Find bounds
    bounds = [(min(i), max(i)) for i in reshaped_lvs]

    # Sample from bounds
    selfies_samples = []
    for j in range(n_samples):
        cur = []
        for i in range(256):
            cur.append(np.random.uniform(bounds[i][0], bounds[i][1]))
        selfies_samples.append(auto.latent_to_smiles(cur, vocab))
        print(j)

    # LV --> SELFIES
    df = pd.DataFrame()
    df['selfies'] = selfies_samples
    df.to_csv(main_path + 'AE_samples.csv')

# Find the euclidean distance between two latent vectors
def vector_distance(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return sum ** (1/2)

def initialize_models(main_path, vocab_path, ae_path, predictor_path, predictor_dataset, suffix=''):
    # Create Vocab
    df = pd.read_csv(vocab_path)
    selfies = list(df['selfies'])
    vocab = Vocabulary(selfies)
    print("Vocab Done!")
    
    # Load AE
    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size

    auto = AE(main_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    print("AE Done!")

    # Load predictor
    property = 'pIC50'
    train_df = pd.read_csv(predictor_dataset)
    predictor = Predictor(predictor_path, property, True, 0.8, vocab, auto, train_df, suffix)
    print("Predictor Done!")

    return vocab, auto, predictor

# Find average pairwise difference between all given latent_vectors
def pairwise_latent_vector_distance(inputs, input_lv, vocab, auto):
    # Get latent vectors    
    if input_lv:
        latent_vectors = inputs
    else:
        tok = vocab.tokenize(inputs)
        encoded = np.array(vocab.encode(tok))
        latent_vectors = auto.sm_to_lat_model.predict(encoded)
    
    # Calculate distances
    distances = []
    for i in range(len(latent_vectors) - 1):
        for j in range(len(latent_vectors) - i - 1):
            distances.append(vector_distance(latent_vectors[i], latent_vectors[i+j+1]))
    return statistics.mean(distances)

# Calculates score of molecules based on QED, SAS, pIC50
# Values are scaled in reference to training dataframe
def molecule_score(samples_df, train_df):
    # Find scale bounds
    qed_vals = list(train_df['QED'])
    sas_vals = list(train_df['SAS'])
    pIC50_vals = list(train_df['pIC50'])
    qed_bounds = (min(qed_vals), max(qed_vals))
    sas_bounds = (min(sas_vals), max(sas_vals))
    pIC50_bounds = (min(pIC50_vals), max(pIC50_vals))

    # Calculate scores
    scores = []
    for i in range(len(samples_df)):
        qed = samples_df.at[i, 'QED']
        sas = samples_df.at[i, 'SAS']
        pIC50 = samples_df.at[i, 'pIC50']
        # Higher pIC50, higher QED, lower SAS --> higher score
        score = (qed-qed_bounds[0])/(qed_bounds[1]-qed_bounds[0]) + (sas_bounds[1]-sas)/(sas_bounds[1]-sas_bounds[0]) + 2 * (pIC50-pIC50_bounds[0])/(pIC50_bounds[1]-pIC50_bounds[0])
        scores.append(score)
    return scores

def molecule_fitness_score(samples_df, train_df):
    # Find scale bounds
    mw_vals = list(train_df['MW'])
    logp_vals = list(train_df['LogP'])
    pIC50_vals = list(train_df['pIC50'])
    mw_bounds = (min(mw_vals), max(mw_vals))
    logp_bounds = (min(logp_vals), max(logp_vals))
    pIC50_bounds = (min(pIC50_vals), max(pIC50_vals))

    # Calculate scores
    scores = []
    for i in range(len(samples_df)):
        mw = samples_df.at[i, 'MW']
        logp = samples_df.at[i, 'LogP']
        pIC50 = samples_df.at[i, 'pIC50']
        # Higher pIC50, lower MW, lower LogP --> higher score
        score = (mw_bounds[1]-mw)/(mw_bounds[1]-mw_bounds[0]) + (logp_bounds[1]-logp)/(logp_bounds[1]-logp_bounds[0]) + (pIC50-pIC50_bounds[0])/(pIC50_bounds[1]-pIC50_bounds[0])
        scores.append(score)
    return scores
    
def rank_all_generations(save_folder, suffix=''):
    main_path = prefix + 'SeawulfGAN/Data/250k_subset.csv'
    train_path = prefix + 'SeawulfGAN/Data/100k_subset_250k.csv'  # Don't change so everything has same reference point for comparison
    ae_path = prefix + 'SeawulfGAN/Models/AE_model.h5'
    predictor_path = prefix + 'SeawulfGAN/Models/'
    predictor_dataset = prefix + 'SeawulfGAN/Data/BACE1.csv'
    
    # Get all files of generated molecules
    all_generations = os.listdir(save_folder)
    all_generations = [g for g in all_generations if 'samples_epoch_' in g]
    
    # Get all samples
    all_samples = []
    for fn in all_generations:
        f = open(save_folder + fn, 'r')
        f.readline()
        samples = f.read().split('\n')
        all_samples.extend(samples)

    # Remove duplicates and invalids and add to dataframe
    all_samples = list(set(all_samples))
    all_smiles = [sf.decoder(se) for se in all_samples]
    all_smiles = [Chem.CanonSmiles(sm) for sm in all_smiles]

    df = pd.DataFrame()
    df['canonical_smiles'] = all_smiles
    df['selfies'] = all_samples
    
    for i in range(len(df)):
        if Chem.MolFromSmiles(df.at[i, 'canonical_smiles']) == None:
            df.drop(labels=i, inplace=True)
    df.drop_duplicates(subset='canonical_smiles', inplace=True)
    df.reset_index(inplace=True)

    # Calculate molecular properties
    df = preprocess.find_molecular_properties(df, None, properties_to_add=['QED', 'SAS'], save=False)
    _, _, predictor = initialize_models(save_folder, main_path, ae_path, predictor_path, predictor_dataset)
    pIC50 = predictor.predict(list(df['selfies']))
    df['pIC50'] = pIC50

    # Calculate fitness
    scores = molecule_score(df, pd.read_csv(train_path))
    df['Score'] = scores
    df.sort_values(by='Score', inplace=True, ascending=False)

    df.to_csv(save_folder + 'all_generations.csv', index=False)

# Check if a molecule is a novel generation, or if it already exists in chembl database
def novelty_check(smiles, all_smiles):
    if smiles in all_smiles:
        return False
    else:
        return True

def percent_novel(df, df_chembl, save_path, checkpoint_path):
    all_smiles = list(df_chembl['canonical_smiles'])

    checkpoints = open(checkpoint_path, 'w')

    total = len(df)
    counter = 0
    novel = []
    for i in range(len(df)):
        if novelty_check(df.at[i, 'canonical_smiles'], all_smiles):
            counter += 1
            novel.append(True)
            checkpoints.write('True\n')
        else:
            novel.append(False)
            checkpoints.write('False\n')
        print(str(i) + " Done")
    print(counter / total)
    checkpoints.write('Percent Novel: ' + str(counter / total))

    # Save
    df['Novel'] = novel
    df.to_csv(save_path, index=False)

def molecule_score_percentage(samples_df, reference_df, train_df, save_path, percentages=[0.01, 0.05, 0.1, 0.25, 0.5]):
    # Ensure that required values are present
    if 'Score' not in samples_df.columns or 'Novel' not in samples_df.columns:
        print('ERROR: Score or Novel values missing from dataframe')
        exit()
    
    ref_scores = molecule_score(reference_df, train_df)
    ref_scores.sort(reverse=True)
    ref_len = len(ref_scores)

    results = open(save_path, 'w')
    for p in percentages:
        cutoff = int(ref_len * p)
        threshold = ref_scores[cutoff-1]

        counter = 0
        for i in range(len(samples_df)):
            if samples_df.at[i, 'Score'] >= threshold:
                if samples_df.at[i, 'Novel']:
                    counter += 1
            else:
                # Since molecules are sorted by score, the remaining molecules also won't meet the threshold
                break

        # Save
        results.write('Percentage: ' + str(p) + '\n')
        results.write('Number: ' + str(counter) + '\n\n')
    
    results.close()

def molecule_fitness_score_percentage(samples_df, reference_df, train_df, save_path, percentages=[0.01, 0.05, 0.1, 0.25, 0.5]):
    # Ensure that required values are present
    if 'Fitness' not in samples_df.columns or 'Novel' not in samples_df.columns:
        print('ERROR: Score or Novel values missing from dataframe')
        exit()

    samples_df.sort_values(by='Fitness', ascending=False, inplace=True, ignore_index=True)
    print(samples_df['Fitness'])
    
    ref_scores = molecule_fitness_score(reference_df, train_df)
    ref_scores.sort(reverse=True)
    ref_len = len(ref_scores)
    print(ref_scores[:10])

    results = open(save_path, 'w')
    for p in percentages:
        cutoff = int(ref_len * p)
        threshold = ref_scores[cutoff-1]

        counter = 0
        for i in range(len(samples_df)):
            if samples_df.at[i, 'Fitness'] >= threshold:
                if samples_df.at[i, 'Novel']:
                    counter += 1
            else:
                # Since molecules are sorted by score, the remaining molecules also won't meet the threshold
                break

        # Save
        results.write('Percentage: ' + str(p) + '\n')
        results.write('Number: ' + str(counter) + '\n\n')
    
    results.close()

def process_known_inhibitors(save_folder):
    names = ['ATABECESTAT', 'VERUBECESTAT', 'LANABECESTAT', 'ELENBECESTAT']
    chembl_ids = ['CHEMBL3916243', 'CHEMBL3301601', 'CHEMBL3989948', 'CHEMBL4204869']

    df = pd.read_csv('Small Molecules/BACE1/BACE1.csv')

    ids = list(df['chembl_id'])
    smiles = list(df['canonical_smiles'])
    for i in range(len(names)):
        index = ids.index(chembl_ids[i])
        sm = smiles[index]
        mol = Chem.MolFromSmiles(sm)
        mol = Chem.rdmolops.AddHs(mol)
        #Draw.MolToFile(mol, save_folder + names[i] + '.png')
        for j in range(50):
            AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=j+1, enforceChirality=True)
            #rdmolfiles.MolToPDBFile(mol, save_folder + names[i] + '_' + str(j+1) + '.pdb')
            rdmolfiles.MolToMolFile(mol, save_folder + names[i] + '_' + str(j+1) + '.mol')

def extract_top_candidate_compounds(df, n, save_folder):
    preparator = MoleculePreparation()
    for i in range(n):
        sm = df.at[i+50, 'canonical_smiles']
        mol = Chem.MolFromSmiles(sm)
        mol = Chem.rdmolops.AddHs(mol)
        for j in range(20):
            AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=j+1, enforceChirality=True)
            rdmolfiles.MolToPDBFile(mol, save_folder + 'MOL' + str(i+51) + '_' + str(j+1) + '.pdb')

def vocab_comparison():
    df = pd.read_csv('Small Molecules/SeawulfGAN/Data/500k_subset.csv')
    smiles = list(df['canonical_smiles'])
    selfies = list(df['selfies'])
    
    vocab_smiles = Vocabulary2(smiles)
    vocab_selfies = Vocabulary(selfies)

    print(vocab_smiles.vocab_size)
    print(vocab_selfies.vocab_size)

    len_smiles = []
    len_selfies = []
    for i in range(len(smiles)):
        len_smiles.append(len(smiles[i]))
        len_selfies.append(len(list(sf.split_selfies(selfies[i]))))

    print(statistics.mean(len_smiles))
    print(statistics.stdev(len_smiles) * 1.96 / math.sqrt(len(smiles)))
    print(statistics.mean(len_selfies))
    print(statistics.stdev(len_selfies) * 1.96 / math.sqrt(len(selfies)))

def distribution_analysis():
    df_gen = pd.read_csv('Small Molecules/GAN Runs/Chembl 100k/Standard/Final Generations/gen_250k.csv')
    df_bace = pd.read_csv('Small Molecules/GAN Runs/BACE1/Standard/Results/gen_250k.csv')

    dfs = [df_gen, df_bace]
    names = ['General', 'Targeted']
    properties = ['pIC50', 'MW', 'LogP']

    results = pd.DataFrame(columns=['Dataset', 'Property', 'Min', 'Max', 'Avg'])
    for i in range(len(dfs)):
        for p in properties:
            vals = list(dfs[i][p])
            if p == 'pIC50':
                vals = [v for v in vals if v >= 0]
            min_val = min(vals)
            max_val = max(vals)
            mean_val = statistics.mean(vals)
            results.loc[len(results)] = [names[i], p, min_val, mean_val, max_val]
    
    results.to_csv('Small Molecules/GAN Runs/Chembl 100k/Standard/comparison.csv', index=False)

def properties_for_molecule_printing(df, molecules, save_file):
    f = open(save_file, 'w')
    for m in molecules:
        index = m - 1
        f.write(str(m) + '\n')
        f.write('pIC50: ' + str(round(df.at[index, 'pIC50'], 2)) + '\t')
        f.write('QED: ' + str(round(df.at[index, 'QED'], 2)) + '\n')
        f.write('SAS: ' + str(round(df.at[index, 'SAS'], 2)) + '\n\n')
    f.close()