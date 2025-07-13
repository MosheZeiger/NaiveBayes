import pandas as pd
import numpy as np
import json
import pprint
import os
from collections import defaultdict


# --- DataPreprocessor Class Definition ---
class DataPreprocessor:
    def __init__(self, index_column_name=None, numeric_columns=None, categorical_columns=None):
        self.index_column_name = index_column_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.fitted_params = {}

    def fit(self, df):
        print("\n[Preprocessor - Fit] Starting to learn preprocessing parameters...")
        
        # Identify numeric and categorical columns if not predefined
        temp_df_for_type_detection = df.drop(columns=[self.index_column_name]) if self.index_column_name and self.index_column_name in df.columns else df
        
        if self.numeric_columns is None:
            self.numeric_columns = temp_df_for_type_detection.select_dtypes(include=np.number).columns.tolist()
        
        if self.categorical_columns is None:
            self.categorical_columns = temp_df_for_type_detection.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Learn means for numeric columns
        for col in self.numeric_columns:
            if col in df.columns:
                self.fitted_params[f'{col}_mean'] = df[col].mean()
        
        print("[Preprocessor - Fit] Finished learning preprocessing parameters.")
        
    def transform(self, df):
        print("\n[Preprocessor - Transform] Starting data transformations and cleaning...")
        processed_df = df.copy()

        # 1. Clean whitespace from all text columns
        for col in processed_df.select_dtypes(include=['object', 'category']).columns:
            processed_df[col] = processed_df[col].astype(str).str.strip()

        # 2. If null appears in the index column, delete the row.
        if self.index_column_name and self.index_column_name in processed_df.columns:
            initial_rows = len(processed_df)
            processed_df.dropna(subset=[self.index_column_name], inplace=True)
            if len(processed_df) < initial_rows:
                print(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} rows due to nulls in index column '{self.index_column_name}'.")

        # 3. If there are duplicates, delete the row.
        initial_rows = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        if len(processed_df) < initial_rows:
            print(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} duplicate rows.")

        # 4. If null appears in a numeric category, impute with the mean.
        if self.numeric_columns:
            for col in self.numeric_columns:
                if col in processed_df.columns:
                    initial_nans = processed_df[col].isnull().sum()
                    if initial_nans > 0:
                        # Fallback if fit was not called (e.g., loading params from file)
                        fill_value = self.fitted_params.get(f'{col}_mean', processed_df[col].mean()) 
                        processed_df[col].fillna(fill_value, inplace=True)
                        print(f"[Preprocessor] Imputed {initial_nans} nulls in numeric column '{col}' with mean: {fill_value:.2f}.")

        # 5. If null appears in a text category, delete the row.
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in processed_df.columns:
                    initial_rows_before_drop_cat_nan = len(processed_df)
                    processed_df.dropna(subset=[col], inplace=True)
                    if len(processed_df) < initial_rows_before_drop_cat_nan:
                        print(f"[Preprocessor] Dropped {initial_rows_before_drop_cat_nan - len(processed_df)} rows due to nulls in categorical column '{col}'.")
        
        print(f"[Preprocessor - Transform] Finished transformations and cleaning. Final rows: {len(processed_df)}")
        return processed_df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save_params(self, file_path):
        # Convert any numpy types to standard Python types for JSON serialization
        serializable_params = {k: v.item() if isinstance(v, np.generic) else v for k, v in self.fitted_params.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
        print(f"[Preprocessor] Preprocessor parameters saved to: {file_path}")

    def load_params(self, file_path):
        with open(file_path, 'r') as f:
            self.fitted_params = json.load(f)
        print(f"[Preprocessor] Preprocessor parameters loaded from: {file_path}")

# --- End of DataPreprocessor Class ---


# --- NaiveBayesClassifier Class Definition ---
class NaiveBayesClassifier:
    def __init__(self, target_column_name, smoothing_alpha=1, preprocessor=None):
        """
        Initializes the NaiveBayesClassifier class.

        :param target_column_name: The name of the target column to predict.
        :param smoothing_alpha: Alpha parameter for Laplace Smoothing. Default is 1.
        :param preprocessor: An instance of DataPreprocessor for preprocessing.
        """
        self.target_column_name = target_column_name
        self.prior_probabilities = {}
        self.likelihood_probabilities = {}
        self.feature_columns = []
        self.target_categories = []
        self.smoothing_alpha = smoothing_alpha
        self.preprocessor = preprocessor

    def _get_feature_columns(self, df, exclude_columns=None):
        """
        Helper function to identify relevant feature columns.
        :param df: The DataFrame from which to extract column names.
        :param exclude_columns: Optional list of columns to exclude from features (e.g., ID columns).
        :return: List of feature column names.
        """
        if exclude_columns is None:
            exclude_columns = []
        
        if self.target_column_name not in exclude_columns:
            exclude_columns.append(self.target_column_name)
            
        return [col for col in df.columns if col not in exclude_columns]

    def _calculate_prior_probabilities(self, df):
        """
        Calculates the prior probabilities of the target column.
        :param df: Processed DataFrame.
        :return: Dictionary of prior probabilities.
        """
        prior_probs = df[self.target_column_name].value_counts(normalize=True).to_dict()
        return prior_probs

    def _calculate_likelihood_probabilities(self, df):
        """
        Calculates likelihood probabilities for each feature and target category.
        Applies Laplace Smoothing.
        :param df: Processed DataFrame.
        :return: Nested dictionary of likelihood probabilities.
        """
        likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for target_category in self.target_categories:
            subset_target = df[df[self.target_column_name] == target_category]
            target_count = len(subset_target)
            
            for feature_col in self.feature_columns:
                feature_unique_values = df[feature_col].unique()
                feature_counts = subset_target[feature_col].value_counts().to_dict()
                
                for feature_value in feature_unique_values:
                    count = feature_counts.get(feature_value, 0)
                    
                    # Laplace Smoothing formula
                    likelihood = (count + self.smoothing_alpha) / (target_count + self.smoothing_alpha * len(feature_unique_values))
                    
                    likelihoods[feature_col][feature_value][target_category] = likelihood
        
        return {k: {vk: dict(vv) for vk, vv in v.items()} for k, v in likelihoods.items()}

    def train(self, train_df_path, excluded_columns_from_features=None):
        """
        Trains the Naive Bayes model.
        The function will perform preprocessing on the raw training file before learning.

        :param train_df_path: Path to the CSV file containing raw training data.
        :param excluded_columns_from_features: Optional list of column names to exclude from features (after preprocessing).
        """
        print(f"\n[Classifier - Train] Starting Naive Bayes model training with file: {train_df_path}")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        # 1. Load raw data
        train_df_raw = pd.read_csv(train_df_path)
        print(f"[Classifier - Train] Loaded {len(train_df_raw)} raw training data rows.")

        # 2. Perform Pre-processing on training data (fit_transform)
        processed_train_df = self.preprocessor.fit_transform(train_df_raw)
        
        # Ensure target column exists after preprocessing
        if self.target_column_name not in processed_train_df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in DataFrame after preprocessing.")

        # 3. Determine feature columns and target categories
        self.feature_columns = self._get_feature_columns(processed_train_df, excluded_columns_from_features)
        self.target_categories = processed_train_df[self.target_column_name].unique().tolist()
        print(f"[Classifier - Train] Feature columns selected for model training: {self.feature_columns}")
        print(f"[Classifier - Train] Target categories: {self.target_categories}")

        # 4. Calculate Prior Probabilities
        self.prior_probabilities = self._calculate_prior_probabilities(processed_train_df)
        print("\n[Classifier - Train] Prior Probabilities:")
        pprint.pprint(self.prior_probabilities)

        # 5. Calculate Likelihood Probabilities
        self.likelihood_probabilities = self._calculate_likelihood_probabilities(processed_train_df)
        print(f"\n[Classifier - Train] Finished calculating likelihood probabilities for {len(self.likelihood_probabilities)} features.")

        print("\n[Classifier - Train] Training process completed successfully.")

    def predict(self, predict_df_path, output_csv_path, excluded_columns_from_features=None, index_column_name=None):
        """
        Performs prediction on new data and saves results to a CSV file.
        The function will perform preprocessing on the raw prediction file before prediction.

        :param predict_df_path: Path to the CSV file containing raw data for prediction.
        :param output_csv_path: Path to the CSV file where results (with predicted target column) will be saved.
        :param excluded_columns_from_features: Optional list of column names to exclude from features.
        :param index_column_name: Name of the index column in the prediction file, if it needs to be included in the output.
        """
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        print(f"\n[Classifier - Predict] Starting prediction for file: {predict_df_path}")
        
        # 1. Load raw prediction data
        predict_df_raw = pd.read_csv(predict_df_path)
        print(f"[Classifier - Predict] Loaded {len(predict_df_raw)} raw prediction data rows.")

        # 2. Perform Pre-processing on prediction data (transform only)
        processed_predict_df = self.preprocessor.transform(predict_df_raw)
        
        # Preserve original index column values if required in output
        original_index_col_values = None
        if index_column_name and index_column_name in processed_predict_df.columns:
            original_index_col_values = processed_predict_df[index_column_name]
            # Temporarily drop index from data used for prediction to prevent it being treated as a feature
            processed_predict_df = processed_predict_df.drop(columns=[index_column_name]) 

        predictions_list = []
        for index, row in processed_predict_df.iterrows():
            posterior_scores = {}
            for target_cat in self.target_categories:
                score = self.prior_probabilities.get(target_cat, 0) # Start with prior probability

                for feature_col in self.feature_columns:
                    feature_value = row[feature_col]

                    # Retrieve likelihood with handling for unseen values
                    likelihood = self.likelihood_probabilities.get(feature_col, {}).get(feature_value, {}).get(target_cat, 
                                (self.smoothing_alpha / (len(self.target_categories) * self.smoothing_alpha))) # Small fallback if something went wrong
                    
                    score *= likelihood
                
                posterior_scores[target_cat] = score
            
            # Normalize probabilities
            total_score = sum(posterior_scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in posterior_scores.items()}
            else: 
                # If all scores are 0 (rare with Laplace smoothing), assign equal probability
                normalized_scores = {k: 1 / len(self.target_categories) for k in self.target_categories} 

            predicted_class = max(normalized_scores, key=normalized_scores.get)
            predictions_list.append(predicted_class)
        
        # Create DataFrame for results and save to CSV
        output_df = pd.DataFrame({self.target_column_name: predictions_list})
        
        # Add back original index column if it was present
        if original_index_col_values is not None:
            # Ensure indices align correctly
            output_df.insert(0, index_column_name, original_index_col_values.reset_index(drop=True)) 
        
        output_df.to_csv(output_csv_path, index=False)
        print(f"[Classifier - Predict] Prediction completed. Results saved to {output_csv_path}")
        return output_df

    def evaluate(self, predict_df_path, actual_results_path, excluded_columns_from_features=None, index_column_name=None):
        """
        Evaluates model performance by comparing predictions to actual results.
        The function will perform preprocessing on the raw prediction file before evaluation.

        :param predict_df_path: Path to the CSV file containing raw data for prediction (same as in predict).
        :param actual_results_path: Path to the CSV file containing the index column and actual target column.
        :param excluded_columns_from_features: Optional list of column names to exclude from features.
        :param index_column_name: Name of the index column in the files.
        :return: Performance report (e.g., accuracy).
        """
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        print(f"\n[Classifier - Evaluate] Starting performance evaluation...")

        # 1. Perform prediction on the test file using the internal predict function
        temp_output_path = "temp_predictions_for_eval.csv"
        predicted_df = self.predict(predict_df_path, temp_output_path, excluded_columns_from_features, index_column_name)
        
        # 2. Load actual results
        actual_df = pd.read_csv(actual_results_path)
        print(f"[Classifier - Evaluate] Loaded {len(actual_df)} rows of actual results.")

        # Ensure index and target columns exist in actual results
        if index_column_name is None or index_column_name not in actual_df.columns:
            raise ValueError(f"Index column '{index_column_name}' not found in actual results file.")
        if self.target_column_name not in actual_df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in actual results file.")

        # 3. Merge predicted and actual results by index column
        evaluation_df = pd.merge(predicted_df, actual_df[[index_column_name, self.target_column_name]], 
                                 on=index_column_name, 
                                 suffixes=('_predicted', '_actual'))
        
        # 4. Calculate accuracy
        correct_predictions = (evaluation_df[f'{self.target_column_name}_predicted'] == evaluation_df[f'{self.target_column_name}_actual']).sum()
        total_predictions = len(evaluation_df)
        
        accuracy = correct_predictions / total_predictions
        
        print(f"\n--- [Classifier - Evaluate] Evaluation Results ---")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Delete temporary output file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"[Classifier - Evaluate] Temporary file {temp_output_path} deleted.")
            
        print("\n[Classifier - Evaluate] Performance evaluation completed.")
        return {"accuracy": accuracy}

# --- End of NaiveBayesClassifier Class ---


# --- Main Script Execution ---
if __name__ == "__main__":
    print("--- Starting Naive Bayes Model Script ---")

    # --- 0. Configurations and Paths ---
    # Ensure these files exist in your script's directory or provide full paths
    TRAIN_RAW_CSV_PATH = 'train.csv' # Path to raw training data CSV
    TEST_RAW_CSV_PATH = 'test.csv'   # Path to raw test/prediction data CSV
    # Path to actual results for evaluation. For Titanic, 'gender_submission.csv' is often provided
    # as a sample submission format, but for true evaluation, you'd need the actual labels for TEST_RAW_CSV_PATH.
    ACTUAL_RESULTS_CSV_PATH = 'gender_submission.csv' 

    PREDICTED_OUTPUT_CSV_PATH = 'titanic_predictions_final.csv'
    PREPROCESSOR_PARAMS_PATH = 'preprocessor_params_final.json'

    TARGET_COL = 'Survived' # Name of your target column
    INDEX_COL = 'PassengerId' # Name of your index column (if any)

    # DataPreprocessor settings (customize for your dataset)
    # It's recommended to define these explicitly for clarity and control.
    PREPROCESSOR_NUMERIC_COLS = ['Age', 'SibSp', 'Parch', 'Fare']
    PREPROCESSOR_CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked'] 

    # Columns to exclude from features for the Naive Bayes Classifier
    # These are columns that might remain in the DataFrame after preprocessing
    # but are not relevant as features for the model (e.g., ID columns).
    EXCLUDED_COLS_FROM_FEATURES_NB = [INDEX_COL] 

    # --- 1. Initialize the Preprocessor ---
    print("\n--- Step 1: Initializing DataPreprocessor ---")
    data_preprocessor = DataPreprocessor(
        index_column_name=INDEX_COL,
        numeric_columns=PREPROCESSOR_NUMERIC_COLS,
        categorical_columns=PREPROCESSOR_CATEGORICAL_COLS
    )

    # --- 2. Initialize the Classifier with the Preprocessor ---
    print("\n--- Step 2: Initializing NaiveBayesClassifier ---")
    classifier = NaiveBayesClassifier(
        target_column_name=TARGET_COL,
        smoothing_alpha=1, # This value can be tuned
        preprocessor=data_preprocessor # Pass the preprocessor instance
    )

    # --- 3. Train the Model (Preprocessor will run internally on raw training data) ---
    print("\n--- Step 3: Training the Model ---")
    try:
        classifier.train(train_df_path=TRAIN_RAW_CSV_PATH, 
                         excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB)
        
        # Optional: Save preprocessor parameters after training
        data_preprocessor.save_params(PREPROCESSOR_PARAMS_PATH)

    except Exception as e:
        print(f"\nError during training: {e}")
        print("Please ensure data files exist and column names in configurations are correct.")
        exit() # Stop script if training fails

    # --- 4. Predict (Preprocessor will run internally on raw prediction data) ---
    print("\n--- Step 4: Making Predictions ---")
    try:
        predicted_df_output = classifier.predict(predict_df_path=TEST_RAW_CSV_PATH, 
                                                output_csv_path=PREDICTED_OUTPUT_CSV_PATH, 
                                                excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB, 
                                                index_column_name=INDEX_COL)
        print(f"\nPredictions saved to {PREDICTED_OUTPUT_CSV_PATH}. Here are the first 5 rows:")
        print(predicted_df_output.head())
    except Exception as e:
        print(f"\nError during prediction: {e}")
        exit() # Stop script if prediction fails

    # --- 5. Evaluate (Preprocessor will run internally) ---
    print("\n--- Step 5: Evaluating Model Performance ---")
    try:
        evaluation_results = classifier.evaluate(predict_df_path=TEST_RAW_CSV_PATH, 
                                                actual_results_path=ACTUAL_RESULTS_CSV_PATH, 
                                                excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB,
                                                index_column_name=INDEX_COL)
        print("\nFinal Evaluation Results:")
        pprint.pprint(evaluation_results)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Please ensure the actual results file contains the correct index and target columns.")
        
    print("\n--- Naive Bayes Model Script Finished ---")