import pandas as pd
import pprint


def load_csv(file_path, file_name):
    df = pd.read_csv(f'{file_path}/{file_name}')
    return df


def calculate_prior_probabilities(df, column):
    total_count = len(df)
    if total_count == 0:
        print("The DataFrame is empty.")
        return 0
    target_counts = (df[column].value_counts() / total_count).round(2)
    return target_counts.to_dict()


def calculate_likelihood_probabilities(df, target_column, not_include_columns=None):
    total_count = len(df)
    if total_count == 0:
        print("The DataFrame is empty.")
        return {}

    feature_columns = []
    for column in df.columns:
        if column != target_column:
            if not_include_columns is None:
                feature_columns.append(column)
            elif isinstance(not_include_columns, list): # Handle multiple columns to exclude
                if column not in not_include_columns:
                    feature_columns.append(column)
            elif column != not_include_columns: # Handle single column to exclude
                feature_columns.append(column)

    likelihoods = {}
    for column in feature_columns:
        likelihoods[column] = {}
        unique_values = df[column].unique()
        for value in unique_values:
            subset = df[df[column] == value]
            if len(subset) == 0:
                continue
            if hasattr(value, "item"):
                clean_value = value.item()
            else:
                clean_value = value
            likelihoods[column][clean_value] = (subset[target_column].value_counts() / len(subset)).round(2).to_dict()
    return likelihoods

# def calculate_likelihood_probabilities(df, target_column):
#     total_count = len(df)
#     if total_count == 0:
#         print("The DataFrame is empty.")
#         return 0
#     feature_columns = [x for x in df.columns if x != target_column]
#     target_likelihoods = list((calculate_prior_probabilities(df, target_column)).keys())
#     likelihoods = {}
#     for column in df.columns:
#         if column != target_column:
#             likelihoods[column] = {}
#             for value in df[column].unique():
#                 print(df[df[column] == value])
#                 # likelihoods[column][value] = (df[df[column] == value][target_column].value_counts() / total_count).round(2)
#     print(likelihoods)
    # return likelihoods
    # return(target_likelihoods)


# log

df = load_csv('.', 'Planned_To_Hike.csv')
# cal_prior_prob = calculate_prior_probabilities(df, 'Planned to Hike?')
cal_likelihood_prob = calculate_likelihood_probabilities(df, 'Planned to Hike?',not_include_columns='Day Number')
pprint.pprint(cal_likelihood_prob)
