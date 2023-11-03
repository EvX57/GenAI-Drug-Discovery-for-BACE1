# GenAI-Drug-Discovery-for-AD
The training datasets, python scripts, trained models, and generated candidate drugs from the *de novo* generative AI drug discovery framework developed in "Discovery of Novel BACE1 Inhibitors for Alzheimer’s Disease with Generative AI."

*`datasets`: contains the preprocessed versions of the general and targeted molecular datasets
*`models`: contains the trained autoencoder, molecular property predictor, WGAN-GP, and GA models
*`candidate drugs`: contains the binding poses of the discovered candidate drugs to the BACE1 active site
*`preprocess.py`: preprocesses the datasets
*`Vocabulary.py`: processes SELFIES strings for the autoencoder model
*`AE.py`: trains and runs the autoencoder model
*`predictor_lv.py`: trains and runs the latent vector molecular property predictor models for pIC50, MW, and Logp prediction
*`WGANGP.py`: implementation of the Wasserstein GAN with Gradient Penalty (WGAN-GP)
*`run.py`: trains and runs the WGAN-GP
*`GA.py`: implementation of the Genetic Algorithm (GA)
*`run_GA.py`: trains and runs the GA
*`create_docking_script.py`: creates a bash script to analyze the compounds in the AutoDock Vina molecular docking simulation
*`docking_analysis.py`: analyzes the results of the docking simulation
*`visualize.py`: various visualization tools for the framework
*`utils.py`: various utility methods for the framework