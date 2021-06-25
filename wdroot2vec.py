import string

import torch
import torchtext

import pandas as pd
import numpy as np


# importing a pre-trained word2vec
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)


# import a word root list with word root and meaning
# can expand to more involved roots data
f_r120 = open('120-root-words-prefixes-and-suffixes.txt')
for i, line in enumerate(f_r120):
    str_line = line.strip()
    if i == 1:
        lst_cols = str_line.split('\t')
        df_r120 = pd.DataFrame(columns=lst_cols)
    elif i > 1:
        lst_line = str_line.split('\t')
        str_rforms = lst_line[0]
        if len(lst_line) > 3:   # python reading '/t' as '\t'
            for j in range(1,len(lst_line)-2):
                str_rforms += '/t' + lst_line[j]
        df_r120.loc[i-2,lst_cols[0]] = str_rforms
        df_r120.iloc[i-2,1] = lst_line[-2]
        df_r120.iloc[i-2,2] = lst_line[-1]
        # each row may contain multiple root forms
f_r120.close()  
# print(df_r120.head())
# print(df_r120.shape)  


# Auxiliary functions
def print_closest_words(vec, n=5):  # copied from https://www.cs.toronto.edu/~lczhang/360/lec/w06/w2v.html
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]: 					       # take the top n
        print(glove.itos[idx], difference)




# structure of methodology:
# The final result should be a set of vector representation of word roots,
# which, at least at a starting point, corresponds to some pre-trained word2vec.
# word2vec and wdroot2vec should be relatable via some kind of mapping,
# and for a starter, I'll try something close to a linear mapping.



# pre-defining parameters
int_vecdim = 50    # dimensions of words and roots are the same
num_roots = df_r120.shape[0]

# method/step 1: correlate word root meaning with word2vec via some linear mapping
# initialize by giving all the words in a root meaning a coefficient of 1
# can adjust the weight based on 1/occurence and such
df_roots = pd.DataFrame(data=np.zeros((num_roots, int_vecdim)))
for i in range(num_roots):
    lst_defwords = df_r120.iloc[i,1].split(' ')
    for wd in lst_defwords:
        df_roots.iloc[i,:] += glove[wd.translate(str.maketrans('', '', string.punctuation))].numpy() / len(lst_defwords)
    tsr_rt = torch.tensor(df_roots.iloc[i,:])
    # print closest words to root vectors
    # print(df_r120.iloc[i,0] + '---' + df_r120.iloc[i,1])
    # print_closest_words(tsr_rt, n=5)



# method/step 2: fit root vectors into the word vector context
# use pre-specified words containing the roots
# initialize with the parameters from step 1, or as a linear combo with coef 1


# method/step 3: determine if a certain letter-sequence is a root with certain meaning


























