from os import path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import selfies as sf
import random

class Vocabulary:
     def __init__(self, all_selfies):
         self.max_len = 0
         self.char_to_int = dict()
         self.int_to_char = dict()
         self.unique_chars = set()
         self.vocab_size = 0
         self.create_vocab(all_selfies)
     
     # Returns list of characters/tokens that make up input SELFIES
     def parse_selfies(self, se):
         return list(sf.split_selfies(se))
    
     def create_vocab(self, all_selfies):
        unique_chars = set()
        for se in all_selfies:
            tokens = self.parse_selfies(se)
             
            if len(tokens) > self.max_len:
                self.max_len = len(tokens)

            for t in tokens:
                unique_chars.add(t)
        # Ensures that even longest SMILES has padding to signal beginning and end of sequence
        self.max_len += 2

         # Padding characters
        unique_chars.add('G')   #start
        unique_chars.add('A')   #padding
        unique_chars = sorted(unique_chars)
        
        vocab = sorted(list(unique_chars))
        self.unique_chars = vocab
        self.vocab_size = len(vocab)

        #encode
        self.char_to_int = dict()
        for i, char in enumerate(vocab):
            self.char_to_int[char] = i
         
        #decode
        self.int_to_char = dict()
        for i, char in enumerate(vocab):
            self.int_to_char[i] = char
     
     # Tokenize SELFIES strings
     def tokenize(self, selfies):
        list_tok_selfies = []
        for se in selfies:
            se_tok = ['G']
            se_tok.extend(self.parse_selfies(se))

            #padding         
            if len(se_tok) < self.max_len:
                 dif = self.max_len - len(se_tok)
                 for _ in range(dif):
                     se_tok.append('A')
                                    
            list_tok_selfies.append(se_tok)

        return list_tok_selfies

     # Encode each char w/ respective index (not one-hot)
     def encode(self, tok_smiles):
         encoded_smiles = []
         for smile in tok_smiles:
             smile_idx = []
             for char in smile:
                 smile_idx.append(self.char_to_int[char])
             encoded_smiles.append(smile_idx)
         return encoded_smiles
     
     # Decode numerical index to respective char
     def decode(self, encoded_smiles):
         smiles = []
         for e_smile in encoded_smiles:
             smile_chars = []
             for idx in e_smile:
                 if (self.int_to_char[idx] == 'G'):
                     continue
                 if (self.int_to_char[idx] == 'A'):
                     break 
                 smile_chars.append(self.int_to_char[idx])
            
             smile_str = ''.join(smile_chars)
         
             smiles.append(smile_str)
         return smiles

     # One hot encode
     def one_hot_encoder(self,smiles_list):
        smiles_one_hot = np.zeros((len(smiles_list),self.max_len, self.vocab_size), dtype = np.int8)
        tokenized = self.tokenize(smiles_list)

        for j, tok in enumerate(tokenized):
           for i, c in enumerate(tok):
               index = self.char_to_int[c]
               smiles_one_hot[j, i, index] = 1

        return smiles_one_hot

     # One hot decode     
     def one_hot_decoder(self, ohe_array):
         all_smiles = []
         for i in range(ohe_array.shape[0]):
             enc_smile = np.argmax(ohe_array[i,: ,:], axis = 1)
             smile = ''
             for i in enc_smile:
                 char = self.int_to_char[i]
                 if char == 'G':
                     continue
                 if char == 'A':
                     break
                 smile += char
             all_smiles.append(smile)
         return all_smiles
    
     # Creates output data
     # Shifts everything in input sequence one earlier
     # So, given input X, you are predicting value of next timestep in output Y
     # This is why you add 'G' token in beginning
     def get_target(self, dataX, encode):
          # For one hot encode
          if encode == 'OHE':
              dataY = np.zeros(shape = dataX.shape, dtype = np.int8)
              for i in range(dataX.shape[0]):
                  # [0:last-1] = [1:last]
                  dataY[i,0:-1,:]= dataX[i, 1:, :]
                  # Add extra padding at end
                  dataY[i,-1,self.char_to_int["A"]] = 1
          # For embedding
          elif encode == 'embedding':   
              dataY = [line[1:] for line in dataX]
              for i in range(len(dataY)):
                 dataY[i].append(self.char_to_int['A'])
     
          return dataY

def test_vocabulary():
    df = pd.read_csv('Small Molecules/Datasets/filtered_chembl_selfies.csv')
    selfies = list(df['selfies'])
    vocab = Vocabulary(selfies)
    
    # Test Period Removal
    if '.' in vocab.unique_chars:
        print('Period Removal Failed')
    else:
        print('Period Removal Successful')

    # Test Vocab Creation
    # Tokens should be 3 longer than selfies alphabet because 'G', 'A'
    length = vocab.vocab_size
    if length == (len(sf.get_alphabet_from_selfies(selfies)) + 2):
        print('Tokenization Successful')
    else:
        print('Tokenization Failed')

    # Test Tokenization, Encode/Decode, OHE/OHD
    samples = selfies[:100]
    tokenized = vocab.tokenize(samples)

    encoded = vocab.encode(tokenized)
    decoded = vocab.decode(encoded)
    if samples == decoded:
        print('Encode-Decode Successful')
    else:
        print('Encode-Decode Failed')
    
    ohe = vocab.one_hot_encoder(samples)
    ohd = vocab.one_hot_decoder(ohe)
    if samples == ohd:
        print('OHE-OHD Successful')
    else:
        print('OHE-OHD Failed')

def test_SELFIES_robustness(save_path, draw=False):
    df = pd.read_csv('Small Molecules/Datasets/filtered_chembl_selfies2.csv')
    selfies = list(df['selfies'])
    vocab = Vocabulary(selfies)
    print('.' in vocab.unique_chars)

    n_samples = 10000
    samples = []
    for _ in range(n_samples):
        cur = []
        for _ in range(vocab.max_len):
            index = random.randint(0, vocab.vocab_size-1)
            cur.append(index)
        samples.append(cur)
    samples = vocab.decode(samples)

    success = 0
    smiles = []
    selfies = []
    for i, s in enumerate(samples):
        sm = sf.decoder(s)
        m = Chem.MolFromSmiles(sm)
        if m != None and len(s) > 0 and len(sm) > 0:
            success += 1
            if draw:
                Draw.MolToFile(m, 'Small Molecules/Random SELFIES/sample_' + str(i) + '.png')
            smiles.append(sm)
            selfies.append(s)

    # Create dataframe
    new_df = pd.DataFrame()
    new_df['canonical_smiles'] = smiles
    new_df['selfies'] = selfies
    new_df.to_csv(save_path)
    
    print(success / n_samples)

def test_vocab_reproducability():
    df = pd.read_csv('Datasets/Properties/250k_subset.csv')
    selfies = list(df['selfies'])
    vocab = Vocabulary(selfies)
    vocab2 = Vocabulary(selfies)
    print(vocab.unique_chars == vocab2.unique_chars)
    print(vocab.char_to_int == vocab2.char_to_int)
    print(vocab.int_to_char == vocab2.int_to_char)