import pandas as pd
import numpy as np
import math
import re
import scipy.stats as stats

def convert_to_snake_case(df):
    #add a space between any lowercase-capital letter pair, then replace spaces with _, the all to lowercase
    new_cols = {col: re.sub(r"([a-z]{1})([A-Z]{1})", r"\1 \2", col).replace(" ", "_").lower() for col in df.columns}
    return df.rename(columns = new_cols, inplace = True)

def zero_counts(df):
    counts = {col: df[col].value_counts().to_dict().get(0) for col in df.columns if df[col].value_counts().to_dict().get(0) != None}
    return {key: value for key, value in counts.items() if value > 0}

def neg_counts(df):
    counts = {col: sum([val for key, val in df[col].value_counts().items() if type(key) in [float, int] and key < 0]) for col in df.columns}
    return {key: value for key, value in counts.items() if value > 0}

def yes_no_to_bin(word):
    true_words = ['yes', 'true', 'y', 't']
    false_words = ['no', 'false', 'n', 'f']

    if str(word).lower() in true_words:
        return 1
    elif str(word).lower() in false_words:
        return 0
    else:
        return word

def get_corr_above_or_below(df, percentage):
    corr = df.corr()
    col_rows = corr.columns.tolist()

    keep_corrs = {}
    for col in col_rows:
        for row in col_rows:
            if corr.loc[col, row] >= percentage or corr.loc[col, row] <= percentage*-1:
                if col == row:
                    break
                if col in keep_corrs:
                    keep_corrs[col][row] = corr.loc[col, row]
                else:
                    keep_corrs[col] = {row: corr.loc[col, row]}

    return keep_corrs

def outlier_dict(df, deviation = 3):
    cols = df.dtypes.to_dict()

    outlier_values = {}
    for col, dtype in cols.items():
        if dtype in [np.float64, np.int64]:
            locations = np.abs(stats.zscore(df[col])) > deviation #credit to: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
            out_rows = df[locations]

            for row in out_rows.iterrows():
                loc = df.index.get_loc(row[0])
                if col in outlier_values:
                    outlier_values[col].append(df.iloc[loc, df.columns.get_loc(col)])
                else:
                    outlier_values[col] = [df.iloc[loc, df.columns.get_loc(col)]]

    for key, val in outlier_values.items():
        outlier_values[key].sort()

    return outlier_values

def corrs_selection(df, col, threshold = .5):
    sale_corr = df.corr().to_dict()[col]
    sale_corr = {key: val for key, val in sale_corr.items() if abs(val) > threshold}

    sale_corr.pop(col)
    return pd.DataFrame(sale_corr, index =[0])



def feature_bucket_candidates(df, threshold = .02):
    candidates = {}

    for col in df.columns.tolist():
        percents = df[col].value_counts(normalize = True, dropna = False).to_dict()

        for key, val in percents.items():
            if df[col].dtypes == np.dtype('O') and val < threshold:
                if col in candidates:
                    candidates[col].append((key, val, threshold))
                else:
                    candidates[col] = [(key, val, threshold)]

    return candidates

def greedy_bucket_selection(bucket_candidates):
    bucket_recommendations = {}

    for col, percents in bucket_candidates.items():
        bucket_recommendations[col] = []
        label_percents = bucket_candidates[col][::-1]
        thresh = label_percents[0][2]

        cumsum = 0
        bucket = []
        for i in range(0, len(label_percents)):
            cumsum += label_percents[i][1]
            bucket.append(label_percents[i][0])

            if cumsum > thresh:
                cumsum = 0
                bucket_recommendations[col].append(bucket)
                bucket = []

        if bucket not in bucket_recommendations[col] and not bucket == []:
            bucket_recommendations[col].append(bucket)

    return bucket_recommendations

def map_buckets(df, buckets):
    for col, bucket_list in buckets.items():
        for recommendation in bucket_list:
            if len(recommendation) > 1:
                strings = map(str, recommendation)
                new_label = '(' + "_".join(strings) + ')'

                df[col] = df[col].map(lambda x: new_label if x in recommendation else x)

    return df

def is_binary_col(series):
    value_dict = series.value_counts().to_dict()
    return series.dtype == np.dtype('int64') and len(value_dict) == 2 and 0 in value_dict and 1 in value_dict

def colinearity_count(df, columns, threshold = .5):
    corr = df.corr()
    counts = {key: 0 for key in columns}

    for row in columns:
        for col in columns:
            if corr[row][col] > threshold and row != col:
                counts[row] += 1

    return counts

def colinearity_pairs(df, columns, threshold = .5):
    corr = df.corr()
    pairs = {key: 0 for key in columns}

    for row in columns:
        pairs[row] = []
        for col in columns:
            if corr[row][col] > threshold and row != col:
                pairs[row].append(col)

    return {col: pair for col, pair in pairs.items() if pair != []}
