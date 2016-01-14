import nltk.data
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# bho: Baltimore housing office permits - lots of stuff
# mpp: Baltimore minor privilege permits (outdoor stuff?)
# rich: Properties for permits for which work is expected to exceed $50,000
# mgc: residential housing permits since 2000 for Montgomery County
bho = pickle.load(open('data/bho.pkl', 'rb'))
mpp = pickle.load(open('data/mpp.pkl', 'rb'))
rich = pickle.load(open('data/rich.pkl', 'rb'))
mgc = pickle.load(open('data/mgc.pkl', 'rb'))

# Can we get the number of project transitions in bho stuff?
ex_v_prop = pd.crosstab(bho['existing_use'], bho['prop_use'])

bho.shape[0] - ex_v_prop.ix['SF', 'SF']
# 246916

(bho.shape[0] - ex_v_prop.ix['SF', 'SF']) / bho.shape[0]
# 0.53644971082693504

(bho.shape[0] - ex_v_prop.ix['SF', 'SF'] -
 bho['existing_use'].value_counts().ix['<Missing>'].values[0] -
 bho['prop_use'].value_counts().ix['<Missing>'].values[0]) / bho.shape[0]
# 0.34315565810227733

ex_v_prop.ix[set(ex_v_prop.columns).difference({'<Missing>'}), set(
    ex_v_prop.columns).difference({'<Missing>'})].sum().sum()
# 415602

# ===
# okay - we want to get a basic plot of all of the residential permits
# over time.

# Messing with descriptions
# It'd be nice to know if there's any meaningful regexes we could
# transform out of the data. (I suspect yes.)


def desc_cleaner(desc):
    return re.sub('[0-9]*(\,?[0-9]*)*\.?[0-9]+', '<NUM>', desc).replace('.<NUM>', '<NUM>')


def get_count_df(descs, tokenizer, preprocessor=desc_cleaner):
    cv = CountVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)
    tfv = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)

    count_counts = cv.fit_transform(descs).sum(0).ravel().tolist()[0]
    tfidf_counts = tfv.fit_transform(descs).sum(0).ravel().tolist()[0]

    vocab_list = [''] * len(cv.vocabulary_)
    for x in cv.vocabulary_:
        vocab_list[cv.vocabulary_[x]] = x

    return pd.DataFrame({'feature': vocab_list,
                         'count': count_counts,
                         'tfidf': tfidf_counts}).sort_values('tfidf', ascending=False)


punkt_sent = nltk.data.load('tokenizers/punkt/english.pickle')

# ===

feature_counts = get_count_df(bho['description'].values, punkt_sent.tokenize)

'seperate permit will be required for TKTK'
'(use) TKTKTK'
'as per code'
'as a single family dwelling'
'TKTK point(s) of protection for low voltage systems'

# ===

feature_counts = get_count_df(mpp['description'].values, punkt_sent.tokenize)

'<NUM> [a-z]* windows'
# In general, pretty pure descriptions. Split on commas, etc.

# ==

feature_counts = get_count_df(rich['description'].values, punkt_sent.tokenize)

'int alts', 'ext alts', 'int/ext alts'
'residential', 'institutional', etc.

feature_counts = get_count_df(
    rich['use_description'].values, punkt_sent.tokenize)

len(set(rich['description']) & set(rich['use_description']))
# 48

# ==

feature_counts = get_count_df(mgc['description'].values, punkt_sent.tokenize)

'deck', 'shed', 'sunroom', 'porch'

# ==

used_cities = set(mgc['city'].ix[mgc['sq_ft'] > -1].value_counts().ix[mgc['city'].ix[mgc['sq_ft'] > -1].value_counts() > 10].reset_index()['index'])

sns.violinplot(x='city', y='sq_ft', data=mgc.ix[(mgc['sq_ft'] > -1) & (mgc['city'].apply(lambda x: x in used_cities))])
