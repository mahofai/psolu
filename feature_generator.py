#####################################
# Create a custom feature generator #
#####################################

from pandas import DataFrame
import torch
from autogluon.features.generators import AbstractFeatureGenerator
from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Feature generator to add k to all values of integer features.
class PlusKFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.
        return X + self.k

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        default_infer_features = dict(valid_raw_types=[R_CATEGORY]) 
        print(default_infer_features)
        return default_infer_features  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


    
class net_charge_Generator(AbstractFeatureGenerator):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.

        def net_charge(seq):
            # Define the pKa values of the amino acids at pH 7.4
            pKa = {'D': 3.9, 'E': 4.3, 'H': 6.0, 'C': 8.3, 'Y': 10.1, 'K': 10.8, 'R': 12.5,
                   'A': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0,
                   'W': 0, 'V': 0}
            # Count the number of each type of amino acid in the sequence
            aa_count = {aa: seq.count(aa) for aa in pKa.keys()}

            # Calculate the net charge of the sequence using the pKa values
            net_charge = sum([-1 * aa_count[aa] * (10 ** (-pKa[aa])) for aa in ['D', 'E']]) \
                         + sum([aa_count[aa] * (10 ** (-pKa[aa])) for aa in ['K', 'R', 'H']]) \
                         + sum([aa_count[aa] for aa in ['C', 'Y', 'K', 'R']])
            return net_charge
    
        print("X:",X)
        df = pd.DataFrame(columns=['net_charge'])
        for column in X.columns:
            df['net_charge'] = X['seq'].apply(net_charge)
        
        return df

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        default_infer_features = dict(valid_raw_types=[R_OBJECT]) 
        print(default_infer_features)
        return default_infer_features  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


class count_charge_Generator(AbstractFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.

        def count_chargeed(seq):
            charged = ['D','E','K','R','H']
            polar = ['S','T','T','Q','C']
            aromatic = ['Y']

            hdrophobic = ['A','V','L','I','M','F','W']
            neutral = ['P','G']

            charged_counter = 0
            polar_counter = 0
            aromatic_counter = 0
            hdrophobic_counter = 0
            neutral_counter = 0

            for c in seq:
                if c in charged:
                    charged_counter+=1
                elif c in polar:
                    polar_counter+=1
                elif c in aromatic:
                    aromatic_counter+=1
                elif c in hdrophobic:
                    hdrophobic_counter+=1
                elif c in neutral:
                    neutral_counter+=1
            return (charged_counter,polar_counter,aromatic_counter,hdrophobic_counter,neutral_counter)

        df = pd.DataFrame(columns=['charged', 'polar', 'aromatic', 'hdrophobic', 'neutral'])
        for column in X.columns:
            df[['charged', 'polar', 'aromatic', 'hdrophobic', 'neutral']] = X[column].apply(count_chargeed).tolist()
        return df

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        default_infer_features = dict(valid_raw_types=[R_OBJECT]) 
        print(default_infer_features)
        return default_infer_features  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.

#TODO feature generator takes max_seq_len as argument  
class one_hot_Generator(AbstractFeatureGenerator):
    # len of max_length seq in train and test dataset
    def __init__(self, seq_type="protein", **kwargs):
        super().__init__(**kwargs)
        self.seq_type = seq_type

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.
        protein_letter_to_int = {'C': 0, 'P': 1, 'R': 2, 'N': 3, 'F': 4, 'K': 5, 'A': 6, 'H': 7, 'Y': 8, 'V': 9, 'L': 10, 'D': 11, 'G': 12, 'E': 13, 'Q': 14, 'M': 15, 'T': 16, 'S': 17, 'I': 18, 'W': 19, 'X':20}
        dna_letter_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        
        letter_to_int = protein_letter_to_int
        if self.seq_type == "protein":
            print("one hot encoding protein!!!!")
            letter_to_int = protein_letter_to_int
        elif self.seq_type == "dna":
            letter_to_int = dna_letter_to_int
            print("one hot encoding dna!!!!")
            
        def one_hot_encoding(sequence,letter_to_int):
            letter_sequence = [letter_to_int[letter] for letter in sequence]
            encoded_tensor  = torch.zeros((len(letter_to_int), max_length), dtype=torch.int64)
            for i in range(len(letter_sequence)):
                encoded_tensor[letter_sequence[i],i] = 1
            return encoded_tensor
        
        # Convert the protein sequences to one-hot encoding
        
        one_hot_df = pd.DataFrame()

        # get the first column 
        column = X.iloc[:, 0]
        sequences = column.tolist()
        max_length = max(len(seq) for seq in sequences)
        one_hot_seqs = []
        for seq in sequences:
            one_hot_seq = one_hot_encoding(seq,letter_to_int)
            one_hot_seqs.append(one_hot_seq.flatten().numpy())

        #print("one_hot_seqs size",one_hot_seqs.shape)
        # Create a dataframe with separate columns for each amino acid position
        column_name = [f'aa{i}_{aa}' for aa in letter_to_int for i in range(1, max_length+1) ]


        #print("one_hot_seqs shape",one_hot_seqs.shape)
        df = pd.DataFrame(one_hot_seqs,columns = column_name)
        df  = df.fillna(value=0).astype("bool")
        return df

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        default_infer_features = dict(valid_raw_types=[R_OBJECT]) 
        print(default_infer_features)
        return default_infer_features  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.

class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G', 'N']
        self.multiplyBy = 5 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 5**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.    
        """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
          seq: 
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

class kmers_Generator(AbstractFeatureGenerator):
    def __init__(self, k=4, write_number_of_occurrences=False, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.write_number_of_occurrences = write_number_of_occurrences
        print("k:",self.k)
        print("write_number_of_occurrences:", write_number_of_occurrences)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.
        column = X.iloc[:, 0]
        seq_list = column.tolist()
        obj = kmer_featurization(self.k)  # initialize a kmer_featurization object
        kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=self.write_number_of_occurrences)
        column_names = [f'kmer{i}' for i in range(5**(self.k))]
        df = pd.DataFrame(kmer_features,columns=column_names)
        return df

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        default_infer_features = dict(valid_raw_types=[R_OBJECT]) 
        print(default_infer_features)
        return default_infer_features  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.

