from collections import Counter
import numpy as np
from Bio import SeqIO
import os


# # Step 1: One-hot encode the DNA sequence
def one_hot_encode(seq):
    mapping = {'A': [1,0,0,0],
               'C': [0,1,0,0],
               'G': [0,0,1,0],
               'T': [0,0,0,1],
               'N': [0,0,0,0]}  # Unknown nucleotide
    return np.array([mapping.get(nuc, [0,0,0,0]) for nuc in seq])

#if needed use this function to convert the sequence to integers
def integer_encode(seq):
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base.upper(), 4) for base in seq]

def read_and_preprocess_data(data):
    all_seqs = []  # empty list to store all sequence
    all_lab = []  # empty list to store all  tpm values
    raw_label = []  # empty list to store all raw 23 tpm values
    with open(f"dataset/{data}.fa", "r") as fasta_file:  # opening the dataset file
        for record in SeqIO.parse(fasta_file, "fasta"):  # assigning each sequence in the name of record
            tpm = record.description.split('|')  # since we have 23 tpm values each separated by | , so to remove | and make list of all 23 tpm values
            tpm.pop(0)  # Since id is the first remove id from description parts
            tpm_values = list(map(float, tpm))  # making the tpm values a proper list and in float
            tpm_mean = np.mean(tpm_values)  # taking mean of 23 tpm values
            all_seqs.append(record.seq)  # append the sequence in the all_seqs
            all_lab.append(tpm_mean)  # append the mean tpm value in the all_label
            raw_label.append(tpm_values)  # Instead of repeating the entire process again to read the labels along with sequence , store the 23 labels separately in a file , for that we are appending the raw 23 tpm values in raw_label

    print(f"Loaded {len(all_seqs)} sequences")  # This line prints How many lines of Dna sequence is read from the dataset

    os.makedirs("data_loader",exist_ok=True)  # this creates the directory data_loader if not already created to store the raw_labels.npy
    np.save("data_loader/Raw_labels.npy", raw_label)

    all_feat = [one_hot_encode(seq) for seq in all_seqs]  # This line converts the raw sequence #ATGC into equivalent one-hot-code
    all_feat = np.array(all_feat)  # Convert the sequence to array
    all_label = np.array(all_lab)  # Converting the labels inton array

    print("Shape of Sequence:", all_feat.shape)  # This line prints the shape of sequence
    print("Shape of labels:", all_label.shape)  # This line prints the shape of Lbels

    # Threshold values to categorize the labels
    low_threshold = 1  #Low threshold below which considered as no expressed
    high_threshold = 6.5  # High threshold above which is considered as highly expressed

    # Function used to categorize the labels for classification task
    def categorize_tpm(tpm):
        if tpm <= low_threshold:
            return 0  # Low
        elif tpm > high_threshold:
            return 2
        else:
            return 1  # High

    labels = np.array([categorize_tpm(tpm) for tpm in all_label])  # This line categorizes all the labels as 0,1,2 . 0 for low,1 for medium and 2 for high

    print("Class distribution:", Counter(labels))  # This line prints how much sequence each class have

    os.makedirs("data_loader",exist_ok=True) # this creates the directory data_loader if not created already to save the sequence and labels
    np.save(f"data_loader/{data}_Features.npy",all_feat)  # save the encoded sequence # one-hot-encoded
    np.save(f"data_loader/{data}_labels.npy", labels)  # Save the categorized labels

    print("Data saved successfully")  # This line indicates the data saved successfully





