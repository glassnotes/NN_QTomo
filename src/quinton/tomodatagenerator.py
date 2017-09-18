#!/usr/bin/python                                                                  
# -*- coding: utf-8 -*-                                                            
#                                                                                  
# tomodatagenerator.py: A class that generates tomography data. 
#                                                                                  
# © 2017 Olivia Di Matteo (odimatte@uwaterloo.ca)                                  
#                                                                                  
# This file is part of the project Quinton.                                      
# Licensed under MIT License.                                                      
# 

import csv

import sys
import qutip as qt
import numpy as np

from multiprocessing import Pool

from quinton.mcexperiment import *
from quinton.eigvecs import *
from quinton.state_utils import *
from quinton.utils import *

class TomoDataGenerator():
    """ A class that generates sample tomography data with which we
        will train a neural network.

        Initialized using the param dictionary.
    """
    def __init__(self, params):
        self.params = params


    def multiproc_generation(self, projs, n_trials, d, mc_engine, bases, op_basis):
        """ On a single processor, generate n_trials random states and frequencies and
            return it for processing by the master thread.
        """
        # Result storage
        input_frequencies = []
        output_coefs = [] 

        for i in range(n_trials):
            state_ket = qt.rand_ket_haar(d)
            state = qt.ket2dm(state_ket).full()

            #freqs = mc_engine.simulate(bases, state)
            freqs = [[float(np.real(np.trace(np.dot(state, projs[basis_idx][j])))) for j in range(len(projs[0]))] for basis_idx in range(len(bases))]

            # Flatten to add to NN data set
            flat_freqs = []
            for s in freqs:
                flat_freqs.extend(s)
            
            # Compute the Bloch vector coefficients using the Gell-Mann basis.
            # This will be a vector of length sqrt((d-1)/2d))
            # The factor of 0.5 is because the traces of the Gell-Mann ops are all 2
            coefs = [0.5 * np.trace(np.dot(x, state)).real for x in op_basis]
            #coefs = extract_parameters(state)

            # Update the data sets
            input_frequencies.append(flat_freqs)
            output_coefs.append(coefs)

        return input_frequencies, output_coefs 


    def generate(self):
        """ Generate training/testing data sets for our neural network, as well as store the
            testing data separately so we can compare it with the LBMLE reconstruction
            algorithm.

            n_trails - Total number of data points
            n_workers - Number of processes to spawn for generation (please make nicely divisible into n_trails)
            percent_test - How much of the data to return as testing data.
            op_basis - The basis expansion in which we should compute the output coefficients
            projs - The projectors of the MUBs in this dimension
            bases - The bases that we're measuring in.

            Returns train_in, train_out, test_in, test_out.
        """
        eigenvectors = None
        d = self.params["DIM"]

        if d == 2:
            eigenvectors = eigenvectors_2
        elif d == 3:
            eigenvectors = eigenvectors_3
        elif d == 4:
            eigenvectors = eigenvectors_4
        elif d == 8:
            eigenvectors = eigenvectors_8
        elif d == 32:
            eigenvectors = eigenvectors_32
        else:
            print("Dimension not supported.")

        # Collect the bases
        bases = []
        if self.params["BASES"] == "all":
            bases = [x for x in range(d)] + [-1]
        else:
            bases = self.params["BASES"]

        # Create the MUBs and MLE simulator
        mc_engine = MCExperiment(eigenvectors)

        projs = generate_projectors(eigenvectors)

        # Set parameters for multiprocessing
        trials_per_worker = int(self.params["N_TRIALS"] * 1./ self.params["N_WORKERS"]) 

        input_params = [(projs, trials_per_worker, d, mc_engine, bases, self.params["OP_BASIS"])] * self.params["N_WORKERS"]

        # Call in the troops!
        pool = Pool(processes = self.params["N_WORKERS"])
        all_data = pool.starmap(self.multiproc_generation, input_params)

        # Unfortunately what comes back is ugly. Thank god for stack overflow!
        # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        all_freqs_ugly = [x[0] for x in all_data]
        train_in = [x for y in all_freqs_ugly for x in y]
        
        all_coefs_ugly = [x[1] for x in all_data]
        train_out = [x for y in all_coefs_ugly for x in y]
             
        with open(self.params["DATA_IN_FILE"], "w") as infile:
            writer = csv.writer(infile)
            writer.writerows(train_in)    

        with open(self.params["DATA_OUT_FILE"], "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(train_out)    
