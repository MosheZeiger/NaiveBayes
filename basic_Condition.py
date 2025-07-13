import pandas as pd
import json
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

def get_feature_columns(df, target_column, not_include_columns=None):
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
    return feature_columns
    

def calculate_likelihood_probabilities(df, target_column, not_include_columns=None):
    total_count = len(df)
    if total_count == 0:
        print("The DataFrame is empty.")
        return {}
    feature_columns = get_feature_columns(df, target_column, not_include_columns)

    likelihoods = {}
    for column in feature_columns:
        likelihoods[column] = {}
        unique_values = df[column].unique()
        for value in unique_values:
            subset = df[df[column] == value]
            if subset.empty:
                continue
            if hasattr(value, "item"):
                clean_value = value.item()
            else:
                clean_value = value
            likelihoods[column][clean_value] = calculate_prior_probabilities(subset, target_column)
            # likelihoods[column][clean_value] = (subset[target_column].value_counts() / len(subset)).round(2).to_dict()
    return likelihoods

def predict_naive_bayes(df, prior_probs, likelihood_probs):
    predictions = []
    for index, row in df.iterrows():
        posterior_probs = {}
        for target_value in prior_probs.keys():
            posterior_probs[target_value] = prior_probs[target_value]
        for column in likelihood_probs.keys():
            if row[column] in likelihood_probs[column]:
                for target_value in posterior_probs:
                    posterior_probs[target_value] *= likelihood_probs[column][row[column]].get(target_value, 0)
        total_prob = sum(posterior_probs.values())
        if total_prob > 0:
            for key in posterior_probs:
                posterior_probs[key] /= total_prob
        # מציאת הערך עם ההסתברות הגבוהה ביותר
        predicted_class = max(posterior_probs, key=posterior_probs.get)
        predictions.append({
            "predicted_class": predicted_class
        })
    return predictions


# log

df_trainer = load_csv('.', 'train.csv')
prior_probs = calculate_prior_probabilities(df_trainer, 'Survived')
likelihood_probs = calculate_likelihood_probabilities(df_trainer, 'Survived',not_include_columns='PassengerId')
df_to_predict = load_csv('.', 'test.csv')
prediction = predict_naive_bayes(df_to_predict, prior_probs, likelihood_probs)
pprint.pprint(prediction)
