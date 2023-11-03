from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate,LSTM, Bidirectional, Dense, Input, GaussianNoise, BatchNormalization, RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
import selfies as sf

from Vocabulary import Vocabulary
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pandas as pd
import random

import time

class Autoencoder:
    def __init__(self, model_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, emb_dim, vocab_size, max_len, write_model_arch=False):
        self.path = model_path 
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.noise_std = noise_std

        self.numb_dec_layer = numb_dec_layer

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.write = write_model_arch

        self.build_smiles_to_latent_model()
        self.build_latent_to_states_model()
        self.build_states_to_smiles_model()
        self.build_model()
    
    # SMILES --> latent representation
    def build_smiles_to_latent_model(self):
        # INPUT: embedded encoding (SHAPE)
        # OUTPUT: latent representation (SHAPE)
        encoder_inputs = Input(shape = (None,), name = 'encoder_inputs')
        x = Embedding(self.vocab_size, self.lstm_units//2)(encoder_inputs)  
        
        states_list = [] 
        states_reversed_list = []
        for i in range(self.numb_dec_layer):
            # Only one layer
            if self.numb_dec_layer == 1:
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Outputs (from both directions), hidden state, cell state, hidden state reversed, cell state reversed
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
            # More than one layer & not last layer
            elif i != self.numb_dec_layer-1:
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_sequences = True, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Outputs (from both directions), hidden state, cell state, hidden state reversed, cell state reversed
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
                
                if self.batch_norm:
                    x  = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i))(x)
            # More than one  layer & last layer
            else:
                encoder2 = Bidirectional(LSTM(self.lstm_units//2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Don't need actual output because it is already captured in hidden state output
                _, state_h2, state_c2, state_h2_reverse, state_c2_reverse = encoder2(x)
                
                states_list.append(state_h2)
                states_list.append(state_c2)
                states_reversed_list.append(state_h2_reverse)
                states_reversed_list.append(state_c2_reverse)
        
        # All hidden and cell states from forward and backward directions from all layers
        complete_states_list = states_list + states_reversed_list
        states = Concatenate(axis = -1, name = 'concatenate')(complete_states_list)

        if self.batch_norm:
            states = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i+1))(states)

        latent_representation = Dense(self.latent_dim, activation = "relu", name = "Dense_relu_latent_rep")(states)

        if self.batch_norm:
            latent_representation = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_latent_rep')(latent_representation)

        #Adding Gaussian Noise as a regularizing step during training
        latent_representation = GaussianNoise(self.noise_std, name = 'Gaussian_Noise')(latent_representation)

        self.smiles_to_latent_model = Model(encoder_inputs, latent_representation, name = 'smiles_to_latent_model')

        if self.write:
            with open(self.path + 'smiles_to_latent.txt', 'w') as f:
                self.smiles_to_latent_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Latent --> initial decoder hidden and cell states for LSTM models
    def build_latent_to_states_model(self):
        # INPUT: latent representation
        # OUTPUT: list w/ 2 elements being 1) hidden states and 2) cell states
        latent_input = Input(shape =(self.latent_dim,), name = 'latent_input')

        #List that will contain the reconstructed states
        decoded_states = []
        for dec_layer in range(self.numb_dec_layer):
            # Hidden and cell states each have a dense layer for reconstruction
            name = "Dense_h_" + str(dec_layer)
            h_decoder = Dense(self.lstm_units, activation = "relu", name = name)(latent_input)

            name = "Dense_c_" + str(dec_layer)
            c_decoder = Dense(self.lstm_units, activation ="relu", name = name)(latent_input)

            if self.batch_norm:
                name = "BN_h_" + str(dec_layer)
                h_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(h_decoder)

                name = "BN_c_" + str(dec_layer)
                c_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(c_decoder)

            decoded_states.append(h_decoder)
            decoded_states.append(c_decoder)

        self.latent_to_states_model = Model(latent_input, decoded_states, name = 'latent_to_states_model')
        if self.write:
            with open(self.path + 'latent_to_states.txt', 'w') as f:
                self.latent_to_states_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Hidden and cell states --> SMILES
    def build_states_to_smiles_model(self):
        # INPUT: hidden and cell states & one hot encoding (teacher forcing)
        # OUTPUT: one hot encoding of predictions (next timestep based on input timesteps)

        #decoder inputs needed for teacher's forcing
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs")

        # One hot + states
        inputs = []
        inputs.append(decoder_inputs)
        x = decoder_inputs

        # Use respective hidden and cell state outputs from encoder layer as input to decoder
        for dec_layer in range(self.numb_dec_layer):
            # Hidden and cell state inputs
            name = "Decoded_state_h_" + str(dec_layer)
            state_h = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_h)

            name = "Decoded_state_c_" + str(dec_layer)
            state_c = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_c)

            #LSTM layer
            decoder_lstm = LSTM(self.lstm_units, return_sequences = True, name = "Decoder_LSTM_" + str(dec_layer))

            x = decoder_lstm(x, initial_state = [state_h, state_c])

            if self.batch_norm:
                x = BatchNormalization(momentum = self.batch_norm_momentum, name = "BN_decoder_"+str(dec_layer))(x)

        #Dense layer that will return probabilities
        outputs = Dense(self.output_dim, activation = "softmax", name = "Decoder_Dense")(x)

        self.states_to_smiles_model = Model(inputs = inputs, outputs = [outputs], name = "states_to_smiles_model")
        if self.write:
            with open(self.path + 'states_to_smiles.txt', 'w') as f:
                self.states_to_smiles_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Combine three components
    def build_model(self):
        encoder_inputs = Input(shape = (None,), name = "encoder_inputs")
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs")
        x = self.smiles_to_latent_model(encoder_inputs)
        x = self.latent_to_states_model(x)
        x = [decoder_inputs] + x
        x = self.states_to_smiles_model(x)

        #Full model
        self.model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = [x], name = "Autoencoder")

    # MAKE SURE THIS WORKS
    def load_autoencoder_model(self, path):
        self.model.load_weights(path)
        #self.model = tf.keras.models.load_model(path)
        self.build_sample_model()
        self.build_sm_to_lat()
    
    # Train
    def fit_model(self, dataX, dataX2, dataY, epochs, batch_size, optimizer):
        self.epochs = epochs
        self.batch_size = batch_size

        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate = 0.001)
        elif optimizer == 'adam_clip':
            self.optimizer = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False, clipvalue = 3)

        ## Callbacks
        checkpoint_dir = self.path 
        checkpoint_file = (checkpoint_dir + "model--{epoch:02d}.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_file, monitor = "val_loss", mode = "min", save_best_only = True)
        
        #Reduces the learning rate by a factor of 2 when no improvement has been see in the validation set for 2 epochs
        reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience=2, min_lr = 1e-6)

        #Early Stopping
        early_stop = EarlyStopping(monitor = "val_loss", patience=5)
        
        callbacks_list = [checkpoint, reduce_lr, early_stop]

        self.model.compile(optimizer  = self.optimizer, loss = 'categorical_crossentropy')
       
        results = self.model.fit([dataX, dataX2], dataY, epochs = self.epochs, batch_size =self.batch_size, validation_split=0.1, shuffle = True, verbose = 1, callbacks = callbacks_list)

        fig, ax = plt.subplots()
        ax.plot(results.history['loss'], label = "Train")
        ax.plot(results.history['val_loss'], label = "Val")
        ax.legend()
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot.png"
        fig.savefig(figure_path)
        plt.close()

        # Build predictor models
        self.build_sample_model()
        self.build_sm_to_lat()

        # Save final models
        self.model.save(self.path + 'AE_model.h5')
        self.sample_model.save(self.path + 'decoder_model.h5')
        self.sm_to_lat_model.save(self.path + 'encoder_model.h5')

    # Convert trained autoencoder into latent --> SMILES model
    def build_sample_model(self):
        # Get the configuration of the batch model
        config = self.states_to_smiles_model.get_config()
        # new_config = config
        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]

        # Remove hidden and cell state inputs
        # States will be directly initialized in LSTM cells for prediction
        idx_list = []
        for idx, layer in enumerate(config["layers"]):
            if "Decoded_state_" in layer["name"]:
                idx_list.append(idx)
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []
            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoded_state_" in inbound_node[0]:
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass
            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (
            1,
            1,
            self.output_dim,
        )

        # Finally, change the statefulness of the LSTM layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True

        # Define the sample_model using the modified config file
        sample_model = Model.from_config(config)

        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in sample_model.layers:
            # Get weights from the batch_model
            weights = self.states_to_smiles_model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            sample_model.get_layer(layer.name).set_weights(weights)

        self.sample_model = sample_model
        return config
    
    # Predict latent --> SMILES
    def latent_to_smiles(self, latent, vocab):
        #predicts the c and h states from the latent representation
        states = self.latent_to_states_model.predict(np.array([latent]))
        
        # Updates the states in the sample model using latent representation
        for dec_layer in range(self.numb_dec_layer): 
            self.sample_model.get_layer("Decoder_LSTM_"+ str(dec_layer)).reset_states(states = [states[2*dec_layer], states[2*dec_layer+1]])
        
        # OHE input
        sample_vector = np.zeros(shape = (1,1,vocab.vocab_size))
        sample_vector[0,0,vocab.char_to_int["G"]] = 1
        smiles = ""
        for i in range(vocab.max_len - 1):
            # Predict character by character, based on previous characters
            pred = self.sample_model.predict(sample_vector)
            idx = np.argmax(pred)
            char = vocab.int_to_char[idx]
            if char == 'G':
                continue
            elif char == 'A':
                break
            else:
                smiles = smiles + char
                sample_vector = np.zeros((1,1,vocab.vocab_size))
                sample_vector[0,0, idx] = 1
        return smiles

    # Convert trained autoencoder into SMILES --> latent model
    def build_sm_to_lat(self):
        # Remove gaussian noise layer
        prediction = self.smiles_to_latent_model.layers[-2].output
        self.sm_to_lat_model = Model(inputs = self.smiles_to_latent_model.input, outputs=prediction)

# Determine percentage of correct molecule reconstruction
def evaluate_reconstruction(real, predicted):
    assert len(real) == len(predicted)
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct = correct+1
    return correct/len(real)*100    

# Determine percentage of characters predicted correctly
def evaluate_reconstruction_partial(real, predicted):
    assert len(real) == len(predicted)
    correct = 0
    total = 0
    for i in range(len(real)):
        index = 0
        while index < len(real[i]) and index < len(predicted[i]):
            if real[i][index] == predicted[i][index]:
                correct += 1
            index += 1
        if len(real[i]) > len(predicted[i]):
            total += len(real[i])
        else:
            total += len(predicted[i])
    return correct / total * 100

# Determine validity of SMILES predictions
# Output: list of valid SMILES and percentage of valid SMILES
def validity(selfies_list):
    total = len(selfies_list)
    valid_selfies = []
    count = 0
    for se in selfies_list:
        sm = sf.decoder(se)
        m = MolFromSmiles(sm)
        if m != None:
            valid_selfies.append(se)
            count += 1
    perc_valid = count/total*100
    
    return valid_selfies, perc_valid

def sample_train_predictions(model, df, vocab, save_path):
    #model = tf.keras.models.load_model('Small Molecules/Sealwulf Runs/Len200/model--60.hdf5')
    #df = pd.read_csv('Small Molecules/Seawulf/10k_chembl_subset.csv')
    smiles = list(df['canonical_smiles'])

    # Randomly sample 100 to test
    num_samples = 1
    random.shuffle(smiles)
    sampled_smiles = smiles[:num_samples]

    # Make predictions
    #vocab = Vocabulary(smiles)
    tok_smiles = vocab.tokenize(sampled_smiles)
    enum_smiles = np.array(vocab.encode(tok_smiles))
    ohe_smiles = vocab.one_hot_encoder(sampled_smiles)
    predicted_smiles_probs = model.predict([enum_smiles, ohe_smiles])

    # Convert softmax OHE --> SMILES
    predicted_smiles = []
    for pred_sm in predicted_smiles_probs:
        cur_smiles = ''
        for ohe_smx in pred_sm:
            index = np.argmax(ohe_smx)
            cur_char = vocab.int_to_char[index]
            if cur_char == 'G':
                continue
            elif cur_char == 'A':
                break
            else:
                cur_smiles += cur_char
        predicted_smiles.append(cur_smiles)
    
    # Write to file
    predictions = open(save_path + 'sample_train_predictions.txt', 'w')
    for i in range(len(predicted_smiles)):
        predictions.write('Actu: ' + sampled_smiles[i])
        predictions.write('\nPred: ' + predicted_smiles[i] + '\n')
    predictions.close()

# Enumerate SMILES
# Takes canonical SMILES as input
def randomize_smiles(smiles, num_randomized, vocab):
    # X vals - randomized
    output = [smiles]
    # Y vals - canonical
    output2 = [smiles for _ in range(num_randomized+1)]

    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    i = 0
    while i < num_randomized:
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        new_smiles = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
        if (new_smiles not in output) and (len(new_smiles) <= vocab.max_len) and (char in vocab.unique_chars for char in new_smiles):
            output.append(new_smiles)
            i += 1
    return output, output2

def initialize():
    main_path = 'Small Molecules/SeawulfGAN/'
    vocab_path = main_path + 'Data/500k_subset.csv'
    ae_path = main_path + 'Models/AE_model_500k.h5'

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

    auto = Autoencoder(main_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    print("Autoencoder Done!")

    return vocab, auto

if __name__ == "__main__" :
    start_time = time.time()

    path = 'path_name'
    filename = 'Data/500k_subset.csv'
    file = path + filename

    run_logistics = open(path + 'logistics.txt', 'w')

    smiles_file = pd.read_csv(file)
    smiles = list(smiles_file['selfies'])
    random.shuffle(smiles)
    
    vocab = Vocabulary(smiles)
    n_train = int(0.8 * len(smiles))
    smiles_train = smiles[:n_train]
    smiles_test = smiles[n_train:]
    
    tok_train = vocab.tokenize(smiles_train)
    tok_test = vocab.tokenize(smiles_test)
    encode_train = np.array(vocab.encode(tok_train))
    encode_test = vocab.encode(tok_test)
    X_train = vocab.one_hot_encoder(smiles_train)
    Y_train = vocab.get_target(X_train, 'OHE')

    run_logistics.write('Vocab Size: ' + str(vocab.vocab_size))
    run_logistics.write('\nMax length: ' + str(vocab.max_len))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    epochs = 200
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = X_train.shape[1:] # = (max_len, vocab.size)
    output_dim = X_train.shape[-1] # = vocab.size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model('Small Molecules/SeawulfGAN/Models/AE_model_500k.h5')
    #auto.fit_model(encode_train, X_train, Y_train, epochs, batch_size, 'adam')

    #run_logistics.write('\nTrain Time: ' + str(round(time.time()-start_time, 3)))

    # Evaluate trained model
    encode_test = encode_test[:500]
    smiles_test = smiles_test[:500]
    latent_vectors = auto.sm_to_lat_model.predict(encode_test)

    predicted_smiles = []
    for lv in latent_vectors:
        predicted_smiles.append(auto.latent_to_smiles(lv, vocab))

    # Save example predictions to file
    example_predictions = open(path + 'sample_test_predictions2.txt', 'w')
    for i in range(len(smiles_test)):
        example_predictions.write('Actu: ' + smiles_test[i])
        example_predictions.write('\nPred: ' + predicted_smiles[i] + '\n')
    example_predictions.close()

    # Calculate statistics
    percent_success = evaluate_reconstruction(smiles_test, predicted_smiles)
    print(percent_success)
    percent_partial_success = evaluate_reconstruction_partial(smiles_test, predicted_smiles)
    print(percent_partial_success)
    _, percent_valid = validity(predicted_smiles)

    test_metrics = open(path + 'results2.txt', 'w')
    test_metrics.write('Percent Total Successful: ' + str(round(percent_success, 4)))
    test_metrics.write('\nPercent Partial Successful: ' + str(round(percent_partial_success, 4)))
    test_metrics.write('\nPercent Valid: ' + str(round(percent_valid, 4)))
    test_metrics.close()

    run_logistics.write('\nTime (seconds): ' + str(round(time.time()-start_time, 3)))
    run_logistics.close()

    # Clean-up and remove saved models from training process
    files = os.listdir(path)
    models = [f for f in files if 'model--' in f]
    epochs = [int(m.split('--')[1].split('.')[0]) for m in models]
    max_epoch = max(epochs)
    for e in epochs:
        if e != max_epoch:
            # Single digit epoch --> add 0 before digit
            if int(e/10) == 0:
                file_name = 'model--0' + str(e) + '.hdf5'
            else:
                file_name = 'model--' + str(e) + '.hdf5'
            os.remove(path + file_name)