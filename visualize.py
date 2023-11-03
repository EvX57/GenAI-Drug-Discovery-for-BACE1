import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import selfies as sf
import os
#import utils
import random
from rdkit.Chem import Draw, MolFromSmiles
import statistics

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def SMILES_length_distribution(smiles, title, save_path):
    lengths = [len(sm) for sm in smiles if len(sm) < 1000]
    sns.displot(data=lengths, kind='hist', kde=True)
    plt.suptitle(title)
    plt.xlabel('Length')
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()

def SELFIES_length_distribution(selfies, title, save_path):
    lengths = []
    for se in selfies:
        length = sf.len_selfies(se)
        if length <= 400:
            lengths.append(length)
    sns.displot(data=lengths, kind='hist', kde=True)
    plt.suptitle(title)
    plt.xlabel('Length')
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()

def compare_property_distributions(dfs, names, x_label, title, property, save_path):
    for i in range(len(dfs)):
        dfs[i][property].plot(kind='kde', label=names[i], xlim=(0, 12))
    plt.title(title)
    plt.xlabel(x_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_property_distribution2(vals, names, x_label, title, save_path):
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#bcbd22', '#17becf']
    fw = 'regular'

    for i in range(len(vals)):
        sns.kdeplot(data=vals[i], fill=True, label=names[i], color=colors[i])
    plt.suptitle(title, fontweight=fw, size=18)
    plt.xlabel(x_label, fontweight=fw, size=14)
    plt.ylabel('Density', fontweight=fw, size=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_all_property_distributions(dfs, names, save_folder):
    properties = ['pIC50', 'QED', 'SAS', 'MW', 'LogP']
    #properties = ['pIC50', 'MW', 'LogP']

    for p in properties:
        vals = []
        for i in range(len(dfs)):
            v = list(dfs[i][p])
            if p == 'pIC50':
                v = [x for x in v if x > 0]
            vals.append(v)
        compare_property_distribution2(vals, names, p, p + ' Distribution', save_folder + p + '_distribution.png')

def validity_progression(path):
    folder = path + 'Generations/'
    all_generations = os.listdir(folder)
    epochs = [int(fn.split('samples_epoch_')[1].split('.txt')[0]) for fn in all_generations]
    sorted_files = [val for (_, val) in sorted(zip(epochs, all_generations), key=lambda x:x[0])]
    epochs.sort()

    percent_valids = []
    for f_name in sorted_files:
        f = open(folder + f_name, 'r')
        line = f.readline()
        percent = float(line.split('Percent Valid: ')[1])
        percent_valids.append(percent)
    
    plt.plot(epochs, percent_valids)
    plt.title('Percent Validity of Generated Molecules')
    plt.xlabel('Epoch')
    plt.ylabel('Percent Valid')
    plt.tight_layout()
    plt.savefig(path + 'percent_valid.png')

# Metrics are internal diversity, uniqueness, latent vector distance
def graph_internal_diversity(save_folder, vocab_path, ae_path, train_df):
    vocab, auto, _ = utils.initialize_models(save_folder, vocab_path, ae_path)
    
    selfies = list(train_df['selfies'])
    random.shuffle(selfies)
    baseline_distance = utils.pairwise_latent_vector_distance(selfies[:1000], False, vocab, auto)
    print("Initialization Complete")
    
    # Get all files and sort by epoch
    all_generations = os.listdir(save_folder)
    all_generations = [g for g in all_generations if 'samples_epoch_' in g]
    epochs = [int(fn.split('samples_epoch_')[1].split('.txt')[0]) for fn in all_generations]
    sorted_files = [val for (_, val) in sorted(zip(epochs, all_generations), key=lambda x:x[0])]
    epochs.sort()

    # Get generatins from each epoch and calculate internal diversity
    int_div = []
    duplicates = []
    distances = []
    counter = 0
    for fn in sorted_files:
        f = open(save_folder + fn, 'r')
        f.readline()
        generations = f.read().split('\n')
        int_div.append(utils.internal_diversity(generations))
        duplicates.append(utils.duplicates(generations))
        distances.append(utils.pairwise_latent_vector_distance(generations, False, vocab, auto) / baseline_distance)
        
        counter += 1
        print(counter)

    # Plot
    plt.plot(epochs, int_div, label='Diversity')
    plt.plot(epochs, duplicates, label='Unique')
    plt.plot(epochs, distances, label='LV Distance')
    plt.title('Epoch vs Internal Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Internal Diversity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'internal_diversity.png')

def compare_molecule_scores(train_df, dfs, names, save_path):
    all_scores = []
    for i in range(len(dfs)):
        if 'Score' not in dfs[i].columns:
            cur_score = utils.molecule_score(dfs[i], train_df)
        else:
            cur_score = list(dfs[i]['Score'])
        # Remove negative scores, because they skew plot
        for i in range(len(cur_score)):
            if cur_score[i] < 0:
                cur_score = cur_score[:i]
                break
        all_scores.append(cur_score)
    
    compare_property_distribution2(all_scores, names, 'Score', 'Score Distributions', save_path)

# Save the best generated and novel molecules to an image
def print_best_molecules(n_mols, df_mols, save_folder):
    i = 0
    counter = 0
    while i < len(df_mols) and counter < n_mols:
        if df_mols.at[i, 'Novel']:
            sm = df_mols.at[i, 'canonical_smiles']
            mol = MolFromSmiles(sm)
            Draw.MolToFile(mol, save_folder + 'molecule_' + str(i+1) + '.png')
            counter += 1
        i += 1

def generation_progression(train_df, folder_path, metric='QED'):
    train_vals = list(train_df[metric])
    train_avg = statistics.mean(train_vals)

    all_files = os.listdir(folder_path)
    all_files = [f for f in all_files if 'samples_epoch_' in f]
    epochs = [int(f.split('samples_epoch_')[1].split('.txt')[0]) for f in all_files]
    all_files = [val for (_, val) in sorted(zip(epochs, all_files), key=lambda x:x[0])]
    epochs.sort()

    # Load predictor model
    if metric == 'pIC50':
        main_path = "/Users/evanxie/Evan's Stuff/Simons 2023/DD/Small Molecules/SeawulfGAN/Models"
        vocab_path = "/Users/evanxie/Evan's Stuff/Simons 2023/DD/Small Molecules/SeawulfGAN/Data/500k_subset.csv"
        ae_path = "/Users/evanxie/Evan's Stuff/Simons 2023/DD/Small Molecules/SeawulfGAN/Models/AE_model_500k.h5"
        predictor_path = "/Users/evanxie/Evan's Stuff/Simons 2023/DD/Small Molecules/SeawulfGAN/Models/"
        predictor_dataset = "/Users/evanxie/Evan's Stuff/Simons 2023/DD/Small Molecules/SeawulfGAN/Data/BACE1.csv"
        suffix = '_500k'
        _, _, predictor = utils.initialize_models(main_path, vocab_path, ae_path, predictor_path, predictor_dataset, suffix)

    average_vals = []
    for i in range(len(all_files)):
        f = open(folder_path+all_files[i], 'r')
        f.readline()
        all_selfies = f.read()
        all_selfies = all_selfies.split('\n')
        if metric == 'QED':
            gen_vals = utils.calculate_qed(all_selfies)
        elif metric == 'pIC50':
            gen_vals = predictor.predict(all_selfies)
        average_vals.append(statistics.mean(gen_vals))

    if metric == 'pIC50':
        i = 0
        while i < len(average_vals):
            if average_vals[i] < 0:
                average_vals.pop(i)
                epochs.pop(i)
                i -= 1
            i += 1

    # Plot
    plt.scatter(epochs, average_vals, label='Generations')
    plt.plot(epochs, [train_avg for _ in epochs], label='Train Data')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title('Average ' + metric + ' vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder_path + metric + '_progression.png')
    plt.close()

def mw_logp_scatter(df, save_path):
    mws = list(df['MW'])
    logps = list(df['LogP'])

    # Sort to inside/outside boundary
    # Boundary: 0<MW<500 & 0<=LogP<=5
    inside = [[] for _ in range(2)]
    outside = [[] for _ in range(2)]
    for i in range(len(mws)):
        if mws[i] > 0 and mws[i] < 500 and logps[i] >= 0 and logps[i] <= 5:
            inside[0].append(mws[i])
            inside[1].append(logps[i])
        else:
            outside[0].append(mws[i])
            outside[1].append(logps[i])
    
    fig, ax = plt.subplots()
    ax.scatter(outside[0], outside[1], s=0.5, color=default_colors[0])
    ax.scatter(inside[0], inside[1], s=0.5, color=default_colors[1])
    ax.add_patch(patches.Rectangle(xy=[0,0], width=500, height=5, linewidth=1, color=default_colors[1], fill=False))
    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.set_title('Generated Compounds')
    plt.savefig(save_path)
    plt.close()

def qed_sas_scatter(df, number, name, save_path):
    df = df.sample(frac=1)
    qed = list(df['QED'])
    sas = list(df['SAS'])
    qed = qed[:number]
    sas = sas[:number]

    max_qed = max(qed)
    min_sas = min(sas)

    # Sort to inside/outside boundary
    # Boundary: 0.5<=QED<=1.0 & 0<=SAS<=5
    inside = [[] for _ in range(2)]
    outside = [[] for _ in range(2)]
    for i in range(len(qed)):
        if qed[i] >= 0.5 and qed[i] <= 1 and sas[i] >= 0 and sas[i] <= 5:
            inside[0].append(qed[i])
            inside[1].append(sas[i])
        else:
            outside[0].append(qed[i])
            outside[1].append(sas[i])
    
    fig, ax = plt.subplots()
    ax.scatter(outside[1], outside[0], s=0.5, color=default_colors[0])
    ax.scatter(inside[1], inside[0], s=0.5, color=default_colors[1])
    ax.add_patch(patches.Rectangle(xy=[0,0.5], width=(5), height=(0.5), linewidth=1, color=default_colors[1], fill=False))
    ax.set_xlabel('SAS')
    ax.set_ylabel('QED')
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.set_title('Generated Molecules')
    plt.savefig(save_path)
    plt.close()

def qed_sas_scatter_2(df, number, save_path):
    df = df.sample(frac=1)
    qed = list(df['QED'])
    sas = list(df['SAS'])
    qed = qed[:number]
    sas = sas[:number]

    # Sort to inside/outside boundary
    # Boundary: 0.5<=QED<=1.0 & 0<=SAS<=5
    inside = [[] for _ in range(2)]
    outside = [[] for _ in range(2)]
    for i in range(len(qed)):
        if qed[i] >= 0.5 and qed[i] <= 1 and sas[i] >= 0 and sas[i] <= 5:
            inside[0].append(qed[i])
            inside[1].append(sas[i])
        else:
            outside[0].append(qed[i])
            outside[1].append(sas[i])
    
    '''fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    sns.scatterplot(x=outside[0], y=outside[1], s=1.5, color=default_colors[0], ax=axs[1,0])
    sns.scatterplot(inside[0], inside[1], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0.5,0], width=0.5, height=5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('QED', fontsize=11)
    axs[1,0].set_ylabel('SAS', fontsize=11)
    #axs[1,0].set_xlabel('QED')
    #axs[1,0].set_ylabel('SAS')
    sns.distplot(qed, hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[4], ax=axs[0,0])
    sns.distplot(sas, hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[4], ax=axs[1,1], vertical=True)
    fig.suptitle('Generated Compounds', fontsize=15, y=0.94)
    plt.savefig(save_path)
    plt.close()'''
    graph_qed_sas_scatter(outside, inside, qed, sas, save_path + '_1.png')
    graph_qed_sas_scatter_2(outside, inside, qed, sas, save_path + '_2.png')
    graph_qed_sas_scatter_3(outside, inside, qed, sas, save_path + '_3.png')
    graph_qed_sas_scatter_4(outside, inside, qed, sas, save_path + '_4.png')

def qed_sas_scatter_3(df, number, save_path):
    df = df.sample(frac=1)
    qed = list(df['QED'])
    sas = list(df['SAS'])
    qed = qed[:number]
    sas = sas[:number]

    # Sort to inside/outside boundary
    # Boundary: 0.5<=QED<=1.0 & 0<=SAS<=5
    inside = [[] for _ in range(2)]
    outside = [[] for _ in range(2)]
    for i in range(len(qed)):
        if qed[i] >= 0.5 and qed[i] <= 1 and sas[i] >= 0 and sas[i] <= 5:
            inside[0].append(qed[i])
            inside[1].append(sas[i])
        else:
            outside[0].append(qed[i])
            outside[1].append(sas[i])
    
    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [5, 1], 'height_ratios': [1, 5]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    sns.scatterplot(x=outside[0], y=outside[1], s=1.5, color=default_colors[0], ax=axs[1,0])
    sns.scatterplot(inside[0], inside[1], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0.5,0], width=0.5, height=5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('QED')
    axs[1,0].set_ylabel('SAS')
    sns.distplot(outside[0], hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[0], ax=axs[0,0])
    sns.distplot(outside[1], hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[0], ax=axs[1,1], vertical=True)
    sns.distplot(inside[0], hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[1], ax=axs[0,0])
    sns.distplot(inside[1], hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[1], ax=axs[1,1], vertical=True)
    #plt.title('Generated Compounds')
    plt.show()
    #plt.savefig(save_path)
    plt.close()

def graph_qed_sas_scatter(outside, inside, qed, sas, save_path):
    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    sns.scatterplot(x=outside[0], y=outside[1], s=1.5, color=default_colors[0], ax=axs[1,0])
    sns.scatterplot(inside[0], inside[1], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0.5,0], width=0.5, height=5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('QED', fontsize=11)
    axs[1,0].set_ylabel('SAS', fontsize=11)
    axs[1,0].set_yticks([0,2,4,6,8])
    axs[1,0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    #axs[1,0].set_xlabel('QED')
    #axs[1,0].set_ylabel('SAS')
    sns.distplot(qed, hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[4], ax=axs[0,0])
    sns.distplot(sas, hist=False, kde=True, kde_kws={"shade":True}, color=default_colors[4], ax=axs[1,1], vertical=True)
    fig.suptitle('Generated Compounds', fontsize=15, y=0.94)
    plt.savefig(save_path)
    plt.close()

def graph_qed_sas_scatter_2(outside, inside, qed, sas, save_path):
    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    sns.scatterplot(x=outside[0], y=outside[1], s=1.5, color=default_colors[0], ax=axs[1,0])
    sns.scatterplot(inside[0], inside[1], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0.5,0], width=0.5, height=5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('QED', fontsize=11, labelpad=7.0)
    axs[1,0].set_ylabel('SAS', fontsize=11, labelpad=7.0)
    axs[1,0].set_yticks([0,2,4,6,8])
    axs[1,0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    #axs[1,0].set_xlabel('QED')
    #axs[1,0].set_ylabel('SAS')
    sns.distplot(qed, hist=True, kde=False, color=default_colors[4], ax=axs[0,0])
    sns.distplot(sas, hist=True, kde=False, color=default_colors[4], ax=axs[1,1], vertical=True)
    fig.suptitle('Discovered Compounds', fontsize=15, y=0.94)
    fig.legend(labels=['All', 'Non-Ideal', 'Ideal'], markerscale=5.0, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.savefig(save_path)
    plt.close()

def graph_qed_sas_scatter_3(outside, inside, qed, sas, save_path):
    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    sns.scatterplot(x=outside[0], y=outside[1], s=1.5, color=default_colors[0], ax=axs[1,0])
    sns.scatterplot(inside[0], inside[1], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0.5,0], width=0.5, height=5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('QED', fontsize=11)
    axs[1,0].set_ylabel('SAS', fontsize=11)
    axs[1,0].set_yticks([0,2,4,6,8])
    axs[1,0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    sns.distplot(qed, hist=True, kde=True, color=default_colors[4], ax=axs[0,0])
    sns.distplot(sas, hist=True, kde=True, color=default_colors[4], ax=axs[1,1], vertical=True)
    fig.suptitle('Generated Compounds', fontsize=15, y=0.94)
    plt.savefig(save_path)
    plt.close()

def graph_qed_sas_scatter_4(outside, inside, qed, sas, save_path):
    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    handle_1 = sns.scatterplot(x=outside[1], y=outside[0], s=1.5, color=default_colors[0], ax=axs[1,0])
    handle_2 = sns.scatterplot(inside[1], inside[0], s=1.5, color=default_colors[1], ax=axs[1,0])
    axs[1,0].add_patch(patches.Rectangle(xy=[0,0.5], width=5, height=0.5, linewidth=1, color=default_colors[1], fill=False))
    axs[1,0].set_xlabel('SAS', fontsize=11, labelpad=7.0)
    axs[1,0].set_ylabel('QED', fontsize=11, labelpad=7.0)
    axs[1,0].set_xticks([0,2,4,6,8])
    axs[1,0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    sns.distplot(sas, hist=True, kde=False, color=default_colors[4], ax=axs[0,0])
    sns.distplot(qed, hist=True, kde=False, color=default_colors[4], ax=axs[1,1], vertical=True)
    fig.suptitle('Discovered Compounds', fontsize=15, y=0.94)
    fig.legend(labels=['All', 'Non-Ideal', 'Ideal'], markerscale=5.0, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.savefig(save_path)
    plt.close()