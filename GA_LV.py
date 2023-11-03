from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Activation, Lambda, Layer, Dropout
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import selfies as sf
import statistics

from time import time
import numpy as np
import pandas as pd
import os
import time

from rdkit.Chem import MolFromSmiles, Descriptors
import random
import matplotlib.pyplot as plt

from WGANGP import WGANGP
from Vocabulary import Vocabulary
from predictor_lv import Predictor
from AE import Autoencoder as AE

class GA(WGANGP):
    def __init__(self, path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters, predictors, df):
        super().__init__(path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters)
        
        self.predictors = predictors

        self.find_scale_bounds(df)

        self.replacement_start_epoch = 0
        self.replacement_percent = 0.1
        self.current_samples = []

    # Find min and max MW, LogP, and pIC50 of training data distribution
    # These bounds are used to calculate fitness function
    def find_scale_bounds(self, df):
        mw = list(df['MW'])
        logp = list(df['LogP'])
        pIC50 = list(df['pIC50'])

        self.scale_bounds_mw = (min(mw), max(mw))
        self.scale_bounds_logp = (min(logp), max(logp))
        self.scale_bounds_pIC50 = (min(pIC50), max(pIC50))

        fitness_vals = self.calculate_fitness([pIC50, mw, logp], lv=False)
        self.scale_bounds_fitness = (min(fitness_vals), max(fitness_vals))

    # Calculate fitness based on MW, LogP, and predicted pIC50 of molecules
    def calculate_fitness(self, samples, lv=True):
        if lv:
            samples = np.array(samples)
            # Calculate pIC50, MW, LogP w/ LV
            pIC50s = self.predictors['pIC50'].predict(samples, string=False)
            MWs = self.predictors['MW'].predict(samples, string=False)
            LogPs = self.predictors['LogP'].predict(samples, string=False)
            # Convert from tensor to list
            pIC50_vals = [p.numpy() for p in pIC50s]
            mw_vals = [m.numpy() for m in MWs]
            logp_vals = [l.numpy() for l in LogPs]
            # Bound mw and logp to > 0
            i = 0
            while i < len(pIC50_vals):
                if mw_vals[i] <= 0 or logp_vals[i] <= 0:
                    # Nullify molecule
                    pIC50_vals[i] = 0
                    mw_vals[i] = 10000
                    logp_vals[i] = 100
                    i -= 1
                i += 1
        else:
            pIC50_vals = samples[0]
            mw_vals = samples[1]
            logp_vals = samples[2]
        
        fitness_vals = []
        for i in range(len(pIC50_vals)):
            mw = mw_vals[i]
            logp = logp_vals[i]
            pIC50 = pIC50_vals[i]

            # Higher fitness for lower mw
            scaled_mw = (self.scale_bounds_mw[1] - mw) / (self.scale_bounds_mw[1] - self.scale_bounds_mw[0])
            # Higher fitness for lower logp
            scaled_logp = (self.scale_bounds_logp[1] - logp) / (self.scale_bounds_logp[1] - self.scale_bounds_logp[0])
            # Higher fitness for higher pIC50
            scaled_pIC50 = (pIC50 - self.scale_bounds_pIC50[0]) / (self.scale_bounds_pIC50[1] - self.scale_bounds_pIC50[0])

            fitness_vals.append(scaled_mw + scaled_logp + scaled_pIC50)
            #fitness_vals.append(scaled_pIC50)

        return fitness_vals

    # Train the critic
    # Includes gradient penalty
    def train_critic(self, x_train):
        # Real samples
        data = x_train
        # Noise from normal distribution for generator
        noise = np.random.uniform(-1,1,(self.batch_size, self.z_dim))

        with tf.GradientTape() as critic_tape:
            self.critic.training = True

            # Generate fake samples
            generated_data = self.generator(noise)

            self.current_samples.extend(generated_data.numpy().tolist())  # REMOVE?
            
            # Evaluate samples w/ critic
            real_output = self.critic(data)
            fake_output = self.critic(generated_data)
            
            # Calculate loss
            critic_loss = K.mean(fake_output) - K.mean(real_output)
            self.critic_loss_real.append(K.mean(real_output))
            self.critic_loss_fake.append(K.mean(fake_output))
            
            # Interpolation b/w real and generated data
            # Part of gradient penalty algorithm
            alpha = tf.random.uniform((self.batch_size,1))
            interpolated_samples = alpha*data +(1-alpha)*generated_data
            
            with tf.GradientTape() as t:
                t.watch(interpolated_samples)
                interpolated_samples_output = self.critic(interpolated_samples)
                
            gradients = t.gradient(interpolated_samples_output, [interpolated_samples])
            
            #computing the Euclidean/L2 Norm
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(gradients_sqr, axis = np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = K.square(1-gradient_l2_norm) #returns the squared distance between L2 norm and 1
            #returns the mean over all the batch samples
            gp =  K.mean(gradient_penalty)
            
            self.gradient_penalty_loss.append(gp)
            # Critic loss
            critic_loss = critic_loss +self.gp_weight*gp
            
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        
        return critic_loss

    # Train the generator
    def train_generator(self):
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        
        with tf.GradientTape() as generator_tape:
            self.generator.training = True
            generated_data = self.generator(noise)
            self.current_samples.extend(generated_data.numpy().tolist())
            
            fake_output = self.critic(generated_data)

            gen_loss = -K.mean(fake_output)
            
        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        return gen_loss

    # Model traning with different selection algorithms
    def train(self, x_train, batch_size, epochs, run_folder, autoencoder, vocab, print_every_n_epochs, train_distribution, critic_loops=5, replace_every_n_epochs=100):        
        self.n_critic = critic_loops
        self.autoencoder = autoencoder
        self.vocab = vocab
        self.replace_every_n_epochs = replace_every_n_epochs

        # Batch and shuffle data
        self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
        
        train_start = time.time()
        
        # Lists to store loss values
        # Loss_Log: time, epoch, loss
        # Loss: loss
        self.g_loss_log = []
        self.critic_loss_log =[]
        self.critic_loss = []
        self.g_loss = []
        self.train_fitness = []
        self.gen_fitness_avg = []
        self.gen_fitness_max = []
        self.gen_fitness_min = []

        for epoch in range(self.epoch, self.epoch+epochs):
            critic_loss_per_batch = []
            g_loss_per_batch = []
            batches_done = 0
            
            for i, batch in enumerate(self.data):
                # Train the critic
                # Trained every batch iteration
                loss_d = self.train_critic(batch)
                critic_loss_per_batch.append(loss_d)
                
                # Train the Generator
                # Trained n_critic batch iterations
                if i % self.n_critic == 0:
                    loss_g = self.train_generator()
                    g_loss_per_batch.append(loss_g)
                    batches_done = batches_done +  self.n_critic
                
                # Save information if it is the last batch ---> end of an epoch
                if i == len(self.data) -1:
                    # Calculate losses for this epoch, based on batch losses
                    self.critic_loss_log.append([time.time()-train_start, epoch, np.mean(critic_loss_per_batch)])
                    self.g_loss_log.append([time.time()-train_start, epoch, np.mean(g_loss_per_batch)])
                    self.critic_loss.append(np.mean(critic_loss_per_batch))
                    self.g_loss.append(np.mean(g_loss_per_batch))		   
                    print( 'Epochs {}: D_loss = {}, G_loss = {}'.format(epoch, self.critic_loss_log[-1][2], self.g_loss_log[-1][2]))

                    # Save information
                    if (epoch % print_every_n_epochs) == 0:
                        print('Saving...')
                        # Save general model information
                        self.save_model(run_folder)
                        self.plot_loss(run_folder)
                        # Save current epoch information
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights.h5'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights.h5'))
                        self.sample_data(200, run_folder, train_distribution, save=True)
                    
                    # Perform genetic operations
                    # Input is selfies representations of molecules
                    if (epoch % replace_every_n_epochs) == 0 and epoch >= self.replacement_start_epoch:
                        x_train = self.replace_elitism(x_train, self.current_samples, run_folder)
                        #x_train = self.replace_roulette(x_train, self.current_samples, run_folder)
                        self.current_samples.clear()

                        # Save new training distribution
                        if (epoch % 50) == 0:
                            df_train_samples = pd.DataFrame()
                            df_train_samples['LV'] = x_train
                            df_train_samples.to_csv(run_folder + 'train_sample_lv.csv')

            self.epoch += 1

    # Take best generated samples and replace with worst
    # Ranking determined by fitness score
    def replace_elitism(self, x_train, generated_samples, run_folder):
        n_replacements = int(len(x_train) * self.replacement_percent)
        n_kept = len(x_train) - n_replacements

        # Highest fitness first, lowest fitness last
        generated_fitness = self.calculate_fitness(generated_samples)
        self.gen_fitness_avg.append(statistics.mean(generated_fitness))
        self.gen_fitness_max.append(max(generated_fitness))
        self.gen_fitness_min.append(min(generated_fitness))
        sorted_gen_samples = [val for (_, val) in sorted(zip(generated_fitness, generated_samples), key=lambda x:x[0], reverse=True)]
        best_gen_samples = sorted_gen_samples[:n_replacements]

        # Find best real samples
        # Highest fitness first, lowest fitness last
        train_fitness = self.calculate_fitness(x_train)
        self.train_fitness.append(statistics.mean(train_fitness))
        sorted_train_samples = [val for (_, val) in sorted(zip(train_fitness, x_train), key=lambda x:x[0], reverse=True)]
        best_train_samples = sorted_train_samples[:n_kept]

        # Combine
        best_train_samples.extend(best_gen_samples)

        # Plot
        self.plot_fitness_progression(run_folder)

        return best_train_samples

    # Randomly replace
    def replace_random(self, x_train_selfies, generated_samples):
        n_replacements = int(len(x_train_selfies) * self.replacement_percent)
        n_kept = len(x_train_selfies) - n_replacements

        # Randomly pick generated samples
        random.shuffle(generated_samples)
        final_gen_samples = generated_samples[:n_replacements]

        # Find best real samples
        # Highest fitness first, lowest fitness last
        train_fitness = self.calculate_fitness(x_train_selfies)
        sorted_train_samples = [val for (_, val) in sorted(zip(train_fitness, x_train_selfies), key=lambda x:x[0], reverse=True)]
        best_train_samples = sorted_train_samples[:n_kept]

        # Combine
        best_train_samples.extend(final_gen_samples)

        return best_train_samples

    # Pick generated samples with probability proportional to their fitness
    def replace_roulette(self, x_train_selfies, generated_samples, run_folder):
        n_replacements = int(len(x_train_selfies) * self.replacement_percent)
        n_kept = len(x_train_selfies) - n_replacements

        # ROULETTE SELECTION
        # Calculate fitnesses & sort accordingly
        generated_fitness = self.calculate_fitness(generated_samples)
        self.gen_fitness_avg.append(statistics.mean(generated_fitness))
        self.gen_fitness_max.append(max(generated_fitness))
        self.gen_fitness_min.append(min(generated_fitness))
        sorted_gen_samples = [val for (_, val) in sorted(zip(generated_fitness, generated_samples), key=lambda x:x[0], reverse=True)]
        generated_fitness.sort(reverse=True)

        # Scale fitnesses
        total_fitness = sum(generated_fitness)
        generated_fitness = [f/total_fitness for f in generated_fitness]

        # Sum fitnesses with each other to create ranges
        for i in range(len(generated_fitness)-1):
            generated_fitness[i+1] = generated_fitness[i] + generated_fitness[i+1]
        generated_fitness.insert(0, 0)
        
        # Sample
        selected_gen_samples = []
        for _ in range(n_replacements):
            random_val = np.random.uniform(0,1)
            for i in range(len(sorted_gen_samples)):
                if random_val >= generated_fitness[i] and random_val < generated_fitness[i+1]:
                    selected_gen_samples.append(sorted_gen_samples[i])
                    break
        
        # Making sure roulette algorithm works
        if len(selected_gen_samples) != n_replacements:
            print("ROULETTE ERROR")

        # Find best real samples
        # Highest fitness first, lowest fitness last
        train_fitness = self.calculate_fitness(x_train_selfies)
        self.train_fitness.append(statistics.mean(train_fitness))
        sorted_train_samples = [val for (_, val) in sorted(zip(train_fitness, x_train_selfies), key=lambda x:x[0], reverse=True)]
        best_train_samples = sorted_train_samples[:n_kept]

        # Combine
        best_train_samples.extend(selected_gen_samples)

        # Plot
        self.plot_fitness_progression(run_folder)

        return best_train_samples

    def plot_fitness_progression(self, run_folder):
        epochs = [(self.replacement_start_epoch + i*self.replace_every_n_epochs) for i in range(len(self.train_fitness))]
        
        # Train samples
        plt.plot(epochs, self.train_fitness)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Train Fitness Progression')
        plt.savefig(run_folder + 'train_fitness_progression.png')
        plt.close()

        # Generated Samples
        # Avg
        plt.plot(epochs, self.gen_fitness_avg)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Gen Fitness Progression (Avg)')
        plt.savefig(run_folder + 'gen_fitness_progression_avg.png')
        plt.close()
        # Max
        plt.plot(epochs, self.gen_fitness_max)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Gen Fitness Progression (Max)')
        plt.savefig(run_folder + 'gen_fitness_progression_max.png')
        plt.close()

if __name__ == '__main__':
    # Paths
    main_path = 'Small Molecules/SeawulfGAN/'
    model_path = main_path + 'Models/'
    vocab_path = main_path + 'Data/250k_subset.csv'
    dataset_path = main_path + 'Data/100k_subset_250k.csv'
    target_dataset_path = main_path + 'Data/BACE1.csv'
    save_path = None

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
    auto.load_autoencoder_model(model_path + 'AE_model.h5')
    print("AE Done!")

    # Load predictor
    property = 'pIC50'
    train_df = pd.read_csv(target_dataset_path)
    predictor = Predictor(model_path, property, True, 0.8, vocab, auto, train_df)
    print("Predictor Done!")

    # Create GAN
    input_dim = 256
    critic_layers_units = [256,256,256]
    critic_lr = 0.0001
    gp_weight = 10
    z_dim  = 64
    generator_layers_units = [128,256,256,256,256]
    generator_batch_norm_momentum = 0.9
    generator_lr = 0.0001
    n_epochs = 1000
    batch_size = 64
    critic_optimizer = 'adam'
    generator_optimizer = 'adam'
    critic_dropout = 0.2
    generator_dropout = 0.2
    n_stag_iters = 50
    print_every_n_epochs = 25
    run_folder = main_path
    critic_path = model_path + 'critic_weights.h5'
    gen_path = model_path + 'generator_weights.h5'
    train_df = pd.read_csv(dataset_path)

    gan = GA(main_path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, generator_optimizer, n_stag_iters, predictor, train_df)
    print("Initialization complete!")
    
    '''
    gen_df = pd.read_csv('GAN Runs (BAD)/Run4/all_generations.csv')
    train_distribution = gan.calculate_fitness(list(train_df['selfies']))
    bace_distribution = gan.calculate_fitness(list(target_df['selfies']))
    #gen_distribution = gan.calculate_fitness(list(gen_df['selfies']))
    #visualize.compare_property_distribution2([train_distribution, bace_distribution, gen_distribution], ['Train (250k)', 'BACE1', 'Generated'], 'Fitness', 'Fitness Distribution', save_path + 'fitness_distribution.png')
    visualize.compare_property_distribution2([train_distribution, bace_distribution], ['Train (250k)', 'BACE1'], 'Fitness', 'Fitness Distribution', 'GA/Fitness Function 1/fitness_distribution_250k.png')'''
