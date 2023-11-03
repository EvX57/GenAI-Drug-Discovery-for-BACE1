import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import random

from Vocabulary import Vocabulary
from AE import Autoencoder

import lazypredict
from lazypredict.Supervised import LazyRegressor

class Predictor():
    def __init__(self, path, property, load, split, vocab, autoencoder, df, suffix='', hyperparams=[1,256,0.0001]):
        self.path = path
        self.property = property
        self.dropout = 0.3
        self.n_layers = hyperparams[0]
        self.n_units = hyperparams[1]
        self.learning_rate = hyperparams[2]
        self.n_epochs = 1000
        self.batch_size = 32
        self.validation_split = 0.1
        self.load = load
        self.split = split
        self.vocab = vocab
        self.auto = autoencoder
        self.data = df
        self.input_length = 256

        if not load:
            self.get_latent_representations()
            self.train_test_split()
        self.build_model()

        if self.load:
            self.load_model(self.property, suffix)
    
    # Convert SMILES to latent vector
    def smiles_to_latentvector(self, smiles):
        '''tokens = []
        for i, sm in enumerate(smiles):
            print(i)
            try:
                tokens.append(self.vocab.tokenize([sm]))
            except AttributeError:
                print(sm)
                exit()'''
        tokens = self.vocab.tokenize(smiles)
        encoded = np.array(self.vocab.encode(tokens))
        latent_vectors = self.auto.sm_to_lat_model.predict(encoded)
        return latent_vectors
    
    # Examine mins and maxs of latent vectors for input scaling
    # Inputs are small enough, don't require scaling
    def examine_latent_vector(self):
        smiles = list(self.data['selfies'])
        latent_vectors = self.smiles_to_latentvector(smiles)

        # Reshape
        reshaped = []
        for j in range(len(latent_vectors[0])):
            cur = []
            for i in range(len(latent_vectors)):
                cur.append(latent_vectors[i][j])
            reshaped.append(cur)

        # Print
        index = [i for i in range(len(reshaped))]
        min_vals = [min(x) for x in reshaped]
        max_vals = [max(x) for x in reshaped]
        new_df = pd.DataFrame()
        new_df['Index'] = index
        new_df['Mins'] = min_vals
        new_df['Maxs'] = max_vals
        new_df.to_csv(self.path + 'mins_maxs.csv')

        print('Min: ' + str(min(min_vals)))
        print('Max: ' + str(max(max_vals)))

    # Add column for latent representations into self.data dataframe
    def get_latent_representations(self):
        smiles = list(self.data['selfies'])
        lat_vecs = self.smiles_to_latentvector(smiles).tolist()
        self.data['LV'] = lat_vecs

    # Create train and test data
    # Input: MACCS fingerprint
    # Output: molecular property
    def train_test_split(self):
        # Shuffle dataframe
        self.data = self.data.sample(frac=1, ignore_index=True)

        # Create X and Y train
        lat_vecs = list(self.data['LV'])
        property = list(self.data[self.property])

        self.range = max(property) - min(property)

        train_length = int(len(lat_vecs) * self.split)
        self.X_train = np.array(lat_vecs[:train_length])
        self.Y_train = np.array(property[:train_length])
        self.X_test = np.array(lat_vecs[train_length:])
        self.Y_test = np.array(property[train_length:])

        # Get input length from latent vector
        self.input_length = len(self.X_train[0])

    # Create model
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_length)))
        for _ in range(self.n_layers):
            model.add(Dense(self.n_units, activation='relu'))
            model.add(Dropout(rate=self.dropout))
        model.add(Dense(1, activation='linear'))

        self.model = model
        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss=MSE, optimizer=opt, metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])

    # Compile and train model          
    def train_model(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
        mc = ModelCheckpoint(self.path + 'best_model_' + self.property + '.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        result = self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks = [es, mc], verbose=0)
        
        # Training curve for MSE
        plt.plot(result.history['loss'], label='Train')
        plt.plot(result.history['val_loss'], label='Validation')
        plt.title('Training Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_loss_' + self.property + '.png')
        plt.close()

        # Training curve for RMSE
        plt.plot(result.history['root_mean_squared_error'], label='Train')
        plt.plot(result.history['val_root_mean_squared_error'], label='Validation')
        plt.title('Training RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_rmse_' + self.property + '.png')
        plt.close()

        # Save model
        self.model.save(self.path + 'model_' + self.property + '.h5')

        print('DONE')

    # Load pre-trained predictor model
    def load_model(self, model_name, suffix=''):
        self.model.load_weights(self.path + "model_" + model_name + suffix + ".h5")
    
    # Evaluate model performance
    # Input
    # X_test: SMILES
    # Y_test: molecular property value
    def evaluate(self):
        performance = self.model.evaluate(self.X_test, self.Y_test)
        mse = performance[0]
        rmse = performance[1]
        mape = performance[2]

        results = open(self.path + 'evaluation_' + self.property + '.txt', 'w')
        results.write('MSE: ' + str(round(mse, 4)))
        results.write('\nRMSE: ' + str(round(rmse, 4)))
        results.write('\nMAPE: ' + str(round(mape, 4)))
        results.close()

        return mse, rmse

    # Make predictions for molecular property
    # Input: SMILES
    # Output: molecular property
    def predict(self, smiles, string=True):
        if string:
            lat_vecs = self.smiles_to_latentvector(smiles)
            predictions = self.model.predict(lat_vecs)
        else:
            lat_vecs = smiles
            predictions = self.model(lat_vecs)
        
        predictions = [p[0] for p in predictions]
        # CHECK
        if tf.is_tensor(predictions[0]):
            predictions = [p.numpy() for p in predictions]
            
        return predictions

    def lazy_model(self, save_folder):
        # Remove quantile regressor (too long)
        all_regressors = lazypredict.Supervised.REGRESSORS
        for i in range(len(all_regressors)):
            if all_regressors[i][0] == 'QuantileRegressor':
                all_regressors.pop(i)
                break
        lazypredict.Supervised.REGRESSORS = all_regressors

        # Train
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, pred = reg.fit(self.X_train, self.X_test, self.Y_train, self.Y_test)

        # Print Results
        results = open(save_folder + 'lazy_reg_results.txt', 'w')
        results.write(pred.to_string(index=False))
        results.write('\nRange: ' + str(self.range))
        results.close()

# Optimize num layers, num hidden units per layer, learning rate, batch size
def optimize_hyperparameters(path, property, df, vocab, auto):
    n_layers = [1, 2, 3]
    n_hidden_units = [128, 256, 512]
    learning_rate = [0.01, 0.001, 0.0001]

    # Num trials for each set of hyperparameters
    # Trials are combined for average and stdev to compare between hyperparameters
    n_trials = 5

    # Run all trials
    counter = 0
    hyperparam_sets = []
    for nl in n_layers:
        for nhu in n_hidden_units:
            for lr in learning_rate:
                predictor = Predictor(path, property, False, 0.8, vocab, auto, df, hyperparams=[nl, nhu, lr])
                trials = []
                for t in range(n_trials):
                    predictor.train_test_split()
                    predictor.build_model()
                    predictor.train_model()
                    mse, _ = predictor.evaluate()
                    trials.append(mse)
                    counter += 1
                    print(str(counter) + ' Done')
                hyperparam_sets.append([nl, nhu, lr, statistics.mean(trials), statistics.stdev(trials)])
    
    # Save results
    new_df = pd.DataFrame(data=hyperparam_sets, columns=['Layers', 'Hidden Units', 'Learning Rate', 'Mean', 'Stdev'])
    new_df.to_csv(path + 'hyperparam_opt.csv', index=False)

def repurpose_for_target(path, property, vocab, auto, df_train, df_repurpose, save_path):
    # Load predictor
    predictor = Predictor(path, property, True, 0.8, vocab, auto, df_train, suffix='_500k')

    # Make predictions
    all_selfies = list(df_repurpose['selfies'])
    #all_selfies = process_lv(df_repurpose)

    print('Predictions starting...')
    predictions = predictor.predict(all_selfies, string=True)
    print('Predictions complete!')

    df_repurpose[property] = predictions
    df_repurpose.to_csv(save_path, index=False)

def process_lv(df):
    lv_strings = list(df['LV'])

    lv_nums = []
    for lv in lv_strings:
        try:
            cur = lv.split(', ')
            for i in range(len(cur)):
                cur[i] = float(cur[i].replace(']', '').replace('[', ''))
            lv_nums.append(np.array(cur, dtype='float32'))
        except ValueError:
            cur = lv.replace(']', '').replace('[', '')
            cur = cur.split()
            for i in range(len(cur)):
                cur[i] = float(cur[i])
            lv_nums.append(np.array(cur, dtype='float32'))

    return np.array(lv_nums)

def test_output_type():
    prefix = 'Small Molecules/'
    path = prefix + 'BACE1/Predictor LV 500k/Run/'
    property = 'pIC50'
    vocab_df = pd.read_csv(prefix + 'SeawulfGAN/Data/500k_subset.csv')
    ae_path = prefix + 'SeawulfGAN/Models/AE_model_500k.h5'
    df = pd.read_csv(prefix + 'SeawulfGAN/Data/BACE1.csv')

    vocab = Vocabulary(list(vocab_df['selfies']))

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
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)

    predictor = Predictor(path, property, True, 0.8, vocab, auto, df, suffix='_500k')

    #input_vals = np.random.random((5, 256))
    input_vals = np.array([np.array([random.randint(0,1) for _ in range(256)]) for _ in range(5)])
    print(predictor.predict(input_vals, string=False))

def run_pIC50():
    prefix = 'Small Molecules/'
    path = prefix + 'BACE1/Predictor LV 500k/Run/'
    property = 'pIC50'
    vocab_df = pd.read_csv(prefix + 'SeawulfGAN/Data/500k_subset.csv')
    ae_path = prefix + 'SeawulfGAN/Models/AE_model_500k.h5'
    df = pd.read_csv(prefix + 'SeawulfGAN/Data/BACE1.csv')

    vocab = Vocabulary(list(vocab_df['selfies']))

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
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k')
    predictor.train_model()
    predictor.evaluate()

    '''df_repurpose = pd.read_csv(data_path)
    save_path = data_path
    print('Started')
    repurpose_for_target(path, property, vocab, auto, df, df_repurpose, save_path)
    print('Done')'''

def run_logp():
    path = 'Small Molecules/BACE1/Predictor LV 500k/Run2/'
    property = 'LogP'
    vocab_df = pd.read_csv('Small Molecules/SeawulfGAN/Data/500k_subset.csv')
    ae_path = 'Small Molecules/SeawulfGAN/Models/AE_model_500k.h5'
    df = pd.read_csv('Small Molecules/SeawulfGAN/Data/500k_subset.csv')

    vocab = Vocabulary(list(vocab_df['selfies']))

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
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.evaluate()

    '''df_repurpose = pd.read_csv(data_path)
    save_path = data_path
    print('Started')
    repurpose_for_target(path, property, vocab, auto, df, df_repurpose, save_path)
    print('Done')'''

def run_MW():
    path = 'Small Molecules/BACE1/Predictor LV 500k/Run3/'
    property = 'MW'
    vocab_df = pd.read_csv('Small Molecules/SeawulfGAN/Data/500k_subset.csv')
    ae_path = 'Small Molecules/SeawulfGAN/Models/AE_model_500k.h5'
    df = pd.read_csv('Small Molecules/SeawulfGAN/Data/500k_subset.csv')

    vocab = Vocabulary(list(vocab_df['selfies']))

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
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df)
    predictor.train_model()
    predictor.evaluate()

    '''df_repurpose = pd.read_csv(data_path)
    save_path = data_path
    print('Started')
    repurpose_for_target(path, property, vocab, auto, df, df_repurpose, save_path)
    print('Done')'''