import string
import random

import torch
import torchtext

import pandas as pd
import numpy as np


# importing a pre-trained word2vec
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)
int_vecdim = list(glove['a'].shape)[0]  # this will be the dimension of all vectors


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
num_roots = df_r120.shape[0]

# method/step 1: correlate word root meaning with word2vec via some linear mapping
# initialize by giving all the words in a root meaning a coefficient of 1
# can adjust the weight based on 1/occurence and such
df_roots = pd.DataFrame(data=np.zeros((num_roots, int_vecdim)), index=df_r120.iloc[:,0])
for i in range(num_roots):
    lst_defwords = df_r120.iloc[i,1].split(' ')
    for wd in lst_defwords:
        df_roots.iloc[i,:] += glove[wd.translate(str.maketrans('', '', string.punctuation))].numpy() / len(lst_defwords)
    ## print closest words to root vectors
    # tsr_rt = torch.tensor(df_roots.iloc[i,:])
    # print(df_r120.iloc[i,0] + '---' + df_r120.iloc[i,1])
    # print_closest_words(tsr_rt, n=5)
df_roots.set_index(df_r120.iloc[:,0])



# method/step 2: fit root vectors into the word vector context
# note that the root vectors and word vectors are unchanged at this stage
# use pre-specified words containing the roots
# initialize with the parameters from step 1, or as a linear combo with coef 1
df_rootdata = df_r120   # df_r120 contains "roots" in the form of 'abc/abd'
col_samplewords = 2
col_root = 0
# record sample words and given roots
df_analyzewords = pd.DataFrame(columns=['word', 'roots', 'rootstarts'])
ra = 0
for ir in range(df_rootdata.shape[0]):  
    str_samplewords = df_rootdata.iloc[ir,col_samplewords]
    lst_samplewords = str_samplewords.split(', ')
    # include root-word relations given by the data
    for sword in lst_samplewords:
        if sword in df_analyzewords.loc[:,'word'].tolist():
            iw = df_analyzewords['word'].loc[df_analyzewords['word']==sword].index.tolist()[0]
            df_analyzewords.loc[iw,'roots'] = df_analyzewords.loc[iw,'roots'] + ', ' + str(df_rootdata.iloc[ir,col_root])
        else:
            df_analyzewords.loc[ra,'word'] = sword
            df_analyzewords.loc[ra,'roots'] = str(df_rootdata.iloc[ir,col_root])
            ra += 1
# set up the equation between roots and words
lst_rootlist = list(df_rootdata.iloc[:,col_root])
df_rwcoef = pd.DataFrame(index=df_analyzewords['word'], columns=lst_rootlist)
df_rwcoef = df_rwcoef.fillna(0)
df_wordsidio = pd.DataFrame(index=df_analyzewords['word'], columns=range(int_vecdim))
df_wordsidio = df_wordsidio.fillna(1)   # initialize the idiosyncratic part of the word at all 1's
for iw in range(df_analyzewords.shape[0]):
    lst_roots = df_analyzewords.loc[iw,'roots'].split(', ')
    for root in lst_roots:
        df_rwcoef.loc[df_rwcoef.index[iw],root] = 1
    int_denom = df_rwcoef.iloc[iw].sum() + 1    # with the idiosyncratic part occupying 1 in the denominator
    df_rwcoef.iloc[iw,:] = df_rwcoef.iloc[iw,:] / int_denom     # with all the coefficients adding up to 1    
    df_wordsidio.iloc[iw,:] = df_wordsidio.iloc[iw,:] / int_denom
    # word = (sum of roots + all 1's) / (number of roots + 1), a linear mapping
# print(df_rwcoef)
# print(df_wordsidio)
# fitting by adjusting the coefficients on the root vectors and the elements of the idiosyncratic vector
max_iter= 100000
pct_neg, pct_multiply, pct_margin = 0.2, 0.3, 0.2
cycle_shakeup, pct_shakeup, mul_shakeup = 7, 0.3, 23
int_nbest = 5
max_idionorm, pct_shrink = 100, 0.008   # shrink sr_idio when it gets too big 
min_idionorm, mul_enlarge = 0.1, 80   # enlarge sr_dio when it gets too small
for sword in df_analyzewords['word']:
    sr_word = pd.Series(data=glove[sword].numpy(),index=range(int_vecdim))
    sr_coef = df_rwcoef.loc[sword,:]
    sr_idio = df_wordsidio.loc[sword,:]
    max_coef = 1 / (len(sr_coef.loc[sr_coef!=0]) + 1) + 1 / (len(sr_coef.loc[sr_coef!=0]) + 1)**2 
    # nbest are not in order of distance
    nbest_coef = pd.DataFrame(index=range(int_nbest), columns=df_rwcoef.columns)
    nbest_idio = pd.DataFrame(index=range(int_nbest), columns=df_wordsidio.columns)
    nbest_loss = pd.Series(index=range(int_nbest)).fillna(-1)
    iter = 0
    while iter < max_iter:
        # "shake" the idiosyncratic vector
        lst_dims = [j for j in range(int_vecdim)]
        if iter % cycle_shakeup == 0:
            idx_exp = random.sample(lst_dims, int(int_vecdim * pct_shakeup / 2))
            sr_idio.loc[idx_exp] = sr_idio.loc[idx_exp] * mul_shakeup
            idx_imp = random.sample(lst_dims, int(int_vecdim * pct_shakeup / 2))
            sr_idio.loc[idx_imp] = sr_idio.loc[idx_imp] / mul_shakeup
        else:
            idx_neg = random.sample(lst_dims, int(int_vecdim * pct_neg))
            sr_idio.loc[idx_neg] = sr_idio.loc[idx_neg] * (-1)
            idx_inc = random.sample(lst_dims, int(int_vecdim * pct_multiply / 2))
            sr_idio.loc[idx_inc] = sr_idio.loc[idx_inc] * (1 + pct_margin)
            idx_dec = random.sample(lst_dims, int(int_vecdim * pct_multiply / 2))
            sr_idio.loc[idx_dec] = sr_idio.loc[idx_dec] * (1 - pct_margin)
        scl_idionorm = sum(sr_idio * sr_idio) ** (1/2.0)
        if scl_idionorm < min_idionorm:
            sr_idio = sr_idio * mul_enlarge
        elif scl_idionorm > max_idionorm:
            sr_idio = sr_idio * pct_shrink
        # adjust the coefficients on the roots
        sr_nonidio = sr_word - sr_idio
        for root in sr_coef.index:
            if sr_coef.loc[root] != 0:
                sum_rtsq = sum(df_roots.loc[root,:] * df_roots.loc[root,:])
                sr_coef.loc[root] = sum(sr_nonidio * df_roots.loc[root,:]) / sum_rtsq**(1/2.0)
        sr_rootportion = df_roots.multiply(sr_coef, axis='index').sum()
        norm_rootportion = sum(sr_rootportion * sr_rootportion) ** (1/2.0)
        norm_nonidio = sum(sr_nonidio * sr_nonidio) ** (1/2.0)
        sr_coef = sr_coef * norm_nonidio / norm_rootportion
        for ic in range(int_vecdim):
            sr_coef.iloc[ic] = min(max_coef, max(-max_coef, sr_coef.iloc[ic]))
        # keep the int_nbest best results, and then go from the worst of kept results
        sr_formulated = df_roots.multiply(sr_coef, axis='index').sum() + sr_idio
        num_distance = sr_word - sr_formulated
        num_distance = sum(num_distance * num_distance) ** (1/2.0)
        for ib in range(int_nbest):
            if nbest_loss.loc[ib] == -1:
                nbest_coef.loc[ib,:] = sr_coef
                nbest_idio.loc[ib,:] = sr_idio
                nbest_loss.loc[ib] = num_distance
                break
            elif num_distance < nbest_loss.loc[ib]:
                print(sr_coef)
                print(sr_idio)
                print(num_distance)
                print(sword)
                idx_max = nbest_loss.idxmax()
                nxt_coef = nbest_coef.loc[idx_max,:]
                nxt_idio = nbest_idio.loc[idx_max,:]
                nbest_coef.loc[idx_max,:] = sr_coef
                nbest_idio.loc[idx_max,:] = sr_idio
                nbest_loss.loc[idx_max] = num_distance
                sr_coef = nxt_coef
                sr_idio = nxt_idio
        iter += 1
        if iter % 50 == 0:
            print(num_distance)
    # record the best result for each word        
    idx_best = nbest_loss.idxmin()        
    df_rwcoef.loc[sword,:] = nbest_coef.iloc[idx_best,:]
    df_wordsidio.loc[sword,:] = nbest_idio.iloc[idx_best,:]
# save results
df_rwcoef.to_csv('df_recoef.txt')
df_wordsidio.to_csv('df_wordsidio.txt')

        
    
    


# method/step 3: determine if a certain letter-sequence is a root with certain meaning
























