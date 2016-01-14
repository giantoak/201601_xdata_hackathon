import glob
import numpy as np
import pandas as pd


def get_first_non_nan(x):
    """Helper function for getting the first non-nan value from a list"""
    try:
        return next(y for y in x if not np.isnan(y))
    except StopIteration:
        return np.nan


def merge_identical_columns(df, cols_to_merge):
    """
    :param list cols_to_merge: list of columns to merge
    :param pandas.DataFrame df: df of columns to merge
    :returns pandas.DataFrame: df with cols_to_merge combined into
    cols_to_merge[0]
    """
    df[cols_to_merge[0]] = df.ix[:, cols_to_merge].apply(get_first_non_nan, 1)
    return df.drop(cols_to_merge[1:], 1)


def main():
    df = pd.concat(pd.read_excel(x) for x in glob.glob('*.xlsx'))
    df.reset_index(drop=True, inplace=True)

    for col_list in [['FORM_  NUMBER', 'FORM_ NUMBER', 'FORM_NUMBER'],
                     ['15_DAY_HOLD?', '15-DAY HOLD'],
                     ['NUMBER OF PAGES', 'NO OF PAGES', '# OF PAGES'],
                     ['CONTRACTOR PHONE', 'CONTRACTORPHONE'],
                     ['PLAN SETS', 'PLANSETS']]:
        df = merge_identical_columns(df, col_list)

    # It seems that a bunch of empty columns were left in one of the form.
    # All of _these_ unnamed columns are entirely empty.
    # There are a few other unnamed columns that contain a few rows of content
    df.drop(['Unnamed: {}'.format(x) for x in range(37, 41)], 1, inplace=True)
    df.drop(['Unnamed: {}'.format(x) for x in range(43, 59)], 1, inplace=True)

    df.to_pickle('sf_merged.pkl')
    df.to_csv('sf_merged.csv', index=False)


if __name__ == "__main__":
    main()
