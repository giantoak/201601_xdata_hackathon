import ujson as json
from itertools import chain
import numpy as np
import pandas as pd
import pickle
import re


def trim_left_zeroes_and_space(x):
    return re.sub('^[0 ]+', '', x).rstrip()

# bho: Baltimore housing office permits - lots of stuff
# mpp: Baltimore minor privilege permits (outdoor stuff?)
# rich: Properties for permits for which work is expected to exceed $50,000
# mgc: residential housing permits since 2000 for Montgomery County

data_path_fmt = 'data/formatted-data/{}.json'.format

pairs = [('mpp', 'bwg6-98m2'),
         ('bho', 'fesm-tgxf'),
         ('rich', 'k6m8-62kn'),
         ('mgc', 'm88u-pqki')]

jsns = {key: json.load(open(data_path_fmt(val)))[val] for key, val in pairs}


def get_facts(jsns):

    # Get counts of shared keys
    # Identical within each set, which isn't too surprising
    pd.Series(list(chain.from_iterable(list((x.keys())
                                            for x in jsns['mgc'])))).value_counts().value_counts()

    key_sets = [set(pd.Series(list(chain.from_iterable(list((x.keys())
                                                            for x in jsns[y])))).value_counts().index) for y in jsns.keys()]

    key_sets[0] & key_sets[1] & key_sets[2]
    # {':created_at',
    #  ':created_meta',
    #  ':id',
    # ':meta',
    # ':position',
    # ':sid',
    # ':updated_at',
    # ':updated_meta'}

    key_sets[0] & key_sets[1] - key_sets[2]
    # set()

    key_sets[0] & key_sets[2] - key_sets[1]
    # {'description', 'location'}

    key_sets[0] & key_sets[2] - key_sets[1] - key_sets[3]
    # {'location'}

    key_sets[0] & key_sets[3] - key_sets[1]
    # {'description'}

    key_sets[0] & key_sets[3] - key_sets[2]
    # set()

    key_sets[1] & key_sets[2] - key_sets[0]
    # {'block', 'lot', 'propertyaddress'}


def get_cleanish_dfs(jsns):
    dfs = {x: pd.DataFrame.from_records(jsns[x]) for x in jsns.keys()}

    # dfs['bho'].columns
    # Index([':created_at', ':created_meta', ':id', ':meta', ':position', ':sid', ':updated_at', ':updated_meta', 'block', 'cost_est', 'dateexpire', 'dateissue', 'existing_use', 'lot', 'permitdescription', 'permitnum', 'prop_use', 'propertyaddress'], dtype='object')

    # permitdescription - trim outer whitespace, rename description
    #  The same as description in mgc? Free text chunks?

    # block - trim leading 0s, make category
    # cost_est - make float
    # dateexpire - make time series, rename date_expire
    # dateissue - make time series, rename date_issue
    # existing_use - make category
    # lot - trim leading 0s, make category
    # permitnum - rename permit_number, make category
    # prop_use - make category
    # propertyaddress - trim leading whitespace and 0s, rename property_address

    cols = ['block', 'lot', 'propertyaddress']
    dfs['bho'].ix[:, cols] = dfs['bho'].ix[:, cols].fillna(
        '<Missing>').applymap(trim_left_zeroes_and_space).apply(pd.Categorical)

    dfs['bho'].ix[:, 'cost_est'] = dfs['bho'].ix[:, 'cost_est'].astype(float)

    cols = ['existing_use', 'prop_use']
    dfs['bho'].ix[:, cols] = dfs['bho'].ix[:, cols].fillna('<Missing>').applymap(
        lambda x: x.strip()).apply(pd.Categorical)

    cols = ['dateexpire', 'dateissue']
    dfs['bho'].ix[:, cols] = dfs['bho'].ix[:, cols].apply(
        lambda x: pd.to_datetime(x, errors='coerce'))

    dfs['bho'].ix[:, 'permitdescription'] = dfs['bho'].ix[:, 'permitdescription'].fillna(
        '').apply(lambda x: ' '.join(x.strip().lower().split()))

    dfs['bho'].rename(columns={'dateexpire': 'date_expire',
                               'dateissue': 'date_issue',
                               'permitdescription': 'description',
                               'permitnum': 'permit_number',
                               'propertyaddress': 'property_address'},
                      inplace=True)

    # ===

    # dfs['mgc'].columns
    # Index([':created_at', ':created_meta', ':id', ':meta', ':position', ':sid',
    #        ':updated_at', ':updated_meta', 'addeddate', 'applicationtype',
    #        'buildingarea', 'city', 'declaredvaluation', 'description',
    #        'finaleddate', 'issueddate', 'location', 'permitno', 'postdir',
    #        'predir', 'state', 'status', 'stname', 'stno', 'suffix', 'usecode',
    #        'worktype', 'zip'],
    #       dtype='object')

    # description - free text - use Refine to find the large, non-unique
    # segments? (alt: chunk into ngrams)

    # addeddate - make time series, rename added_date
    # applicationtype - make category, rename application_type, drop? (only one value)
    # buildingarea - rename building_area, deal with decimals?
    # city - strip, make category
    # declaredvaluation - make float (or int) rename declared_valuation
    # finaleddate - make time series, rename... something
    # issueddate - make time series, rename issued_date
    # location - pull out human_address: address as property_address,
    #   human_address:city as city (skipped), human_address:state as
    #   state (skipped), and human_address:zip as zip (skipped)
    # permitno - rename permit_number
    # postdir - make category, rename post_dir
    # predir - make category, rename pre_dir
    # state - make category
    # stname - make category, rename street_name
    # stno - make category, rename street_number
    # usecode - make category, rename use_code
    # worktype - make category, rename work_type
    # zip - make category

    del dfs['mgc']['applicationtype']

    cols = ['addeddate', 'finaleddate', 'issueddate']
    dfs['mgc'].ix[:, cols] = dfs['mgc'].ix[:, cols].apply(pd.to_datetime)

    cols = ['city', 'postdir', 'predir', 'state',
            'stname', 'usecode', 'worktype', 'zip']
    dfs['mgc'].ix[:, cols] = dfs['mgc'].ix[:, cols].fillna('<Missing>').applymap(
        lambda x: x.strip()).apply(pd.Categorical)

    dfs['mgc'].ix[:, 'stno'] = dfs['mgc'].ix[:, 'stno'].fillna('').apply(
        trim_left_zeroes_and_space).astype('category')

    cols = ['buildingarea', 'declaredvaluation']
    dfs['mgc'].ix[:, cols] = dfs['mgc'].ix[:, cols].astype(float)

    dfs['mgc']['property_address'] = dfs['mgc'][
        'location'].apply(lambda x: x['human_address'])

    dfs['mgc'].ix[:, 'description'] = dfs['mgc'].ix[:, 'description'].fillna(
        '').apply(lambda x: ' '.join(x.strip().lower().split()))

    dfs['mgc'].rename(columns={'addeddate': 'added_date',
                               'applicationtype': 'application_type',
                               'buildingarea': 'building_area',
                               'declaredvaluation': 'declared_valuation',
                               'finaleddate': 'finaled_date',
                               'issueddate': 'issued_date',
                               'permitno': 'permit_number',
                               'postdir': 'post_dir',
                               'predir': 'pre_dir',
                               'stname': 'street_name',
                               'stno': 'street_number',
                               'usecode': 'use_code',
                               'worktype': 'work_type'},
                      inplace=True)

    # ===

    dfs['rich'].columns
    # Index([':created_at', ':created_meta', ':id', ':meta', ':position', ':sid',
    #        ':updated_at', ':updated_meta', 'amount', 'description', 'issuedate',
    #        'location_1', 'tract', 'type', 'usedescription'],
    #       dtype='object')

    # description - free text - use Refine to find the large, non-unique
    # segments? (alt: chunk into ngrams)

    # amount - rename cost_est?
    # issuedate - make time series, rename issue_date? (Not for now)
    # tract - make category
    # location_1 - pull out human_address: address as property_address, human_address: city as city, human_address: state as state, human_address: zip as zip, rename location
    # usedescription - rename use_description
    # type - merge duplicate labels, make category, compare

    cols = ['amount']
    dfs['rich'].ix[:, cols] = dfs['rich'].ix[:, cols].astype(float)

    dfs['rich'].ix[:, 'issuedate'] = dfs['rich'].ix[
        :, 'issuedate'].apply(pd.to_datetime)

    dfs['rich'].ix[:, 'type'] = dfs['rich'].ix[:, 'type'].apply(lambda x: x.replace(
        '- Mixed Permit Type', '').split('-')[0].strip()).astype('category')

    for val in ['city', 'state', 'zip']:
        dfs['rich'][val] = dfs['rich']['location_1'].apply(lambda x: x[val] if val in x else '<Missing>')

    cols = ['tract'] + ['city', 'state', 'zip']
    dfs['rich'].ix[:, cols] = dfs['rich'].ix[:, cols].fillna('<Missing>').applymap(
        lambda x: x.strip()).apply(pd.Categorical)

    dfs['rich']['property_address'] = dfs['rich'][
        'location_1'].apply(lambda x: x['human_address'])

    for col in ['description', 'usedescription']:
        dfs['rich'].ix[:, col] = dfs['rich'].ix[:, col].fillna(
        '').apply(lambda x: ' '.join(x.strip().lower().split()))

    dfs['rich'].rename(columns={'issuedate': 'issue_date',
                                'location_1': 'location',
                                'usedescription': 'use_description'},
                       inplace=True)

    # ===

    dfs['mpp'].columns
    # Index([':created_at', ':created_meta', ':id', ':meta', ':position', ':sid',
    #        ':updated_at', ':updated_meta', 'block', 'description', 'location',
    #        'lot', 'permit_number', 'propertyaddress'],
    #       dtype='object')

    # description - clean up text
    # location - pull out human_address: address as property_address (compare
    # with propertyaddress), human_address: city, human_address: state

    # block - strip leading zerioes (unless aadress is 0), make category
    # lot - make category
    # permit_number
    # propertyaddress - rename property_address

    for val in ['city', 'state', 'zip']:
        dfs['mpp'][val] = dfs['rich']['location'].apply(lambda x: x[val] if val in x else '<Missing>')

    cols = ['block', 'lot'] + ['city', 'state', 'zip']
    dfs['mpp'].ix[:, cols] = dfs['mpp'].ix[:, cols].fillna('<Missing>').applymap(
        lambda x: x.strip()).apply(pd.Categorical)

    dfs['mpp'].ix[:, 'description'] = dfs['mpp'].ix[:, 'description'].fillna(
        '').apply(lambda x: ' '.join(x.strip().lower().split()))

    dfs['mpp'].rename(columns={'propertyaddress': 'property_address'},
                      inplace=True)

    # ===

    return dfs


# Do _something_ intelligent with the description fields

# bring dfs['mgc']['work_type'] and dfs['rich']['type'] into alignment
# Align dfs['rich']['use_description'] into alignment

def main():
    data_path_fmt = 'data/formatted-data/{}.json'.format

    pairs = [('mpp', 'bwg6-98m2'),
             ('bho', 'fesm-tgxf'),
             ('rich', 'k6m8-62kn'),
             ('mgc', 'm88u-pqki')]

    jsns = {key: json.load(open(data_path_fmt(val)))[
        val] for key, val in pairs}

    # Various stat things
    get_facts(jsns)

    dfs = get_cleanish_dfs(jsns)

    with open('data/md_dfs.pkl', 'wb') as outfile:
        pickle.dump(dfs, outfile, 2)

if __name__ == "__main__":
    main()
