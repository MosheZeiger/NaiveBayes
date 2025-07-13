import pandas as pd
import numpy as np
import json
import os
import logging
from collections import defaultdict
from datetime import datetime


def ensure_directory_exists(path):
    # Determine if the path is a file path or a directory path
    # If it looks like a file (has a base name and a dot), get its directory
    if os.path.basename(path) and '.' in os.path.basename(path):
        directory_path = os.path.dirname(path)
    else:
        directory_path = path

    # Handle cases where directory_path might be empty (e.g., for just 'my_file.txt')
    if not directory_path:
        directory_path = '.' # Default to current directory

    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            # We can't use 'logger' yet here, as logger might not be fully configured if this is called first.
            # print(f"Directory created: {directory_path}") # For initial setup feedback
            # For a fully configured logger, you'd use logger.info(), but it depends on call order.
            # For now, let's assume logging.basicConfig is called *after* this directory is ensured.
            pass # Just create it silently here, the main logger will confirm later if needed.
        except OSError as e:
            # If an error occurs *here*, before the logger is fully set up, we'll print.
            # In a real app, you'd want robust error handling.
            print(f"ERROR: Could not create directory {directory_path}: {e}")
            raise # Re-raise the exception as directory creation is critical


# --- Logger Configuration ---
# Set up a logger that writes to a file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-4]
LOG_FILE_PATH = f'./log/model_run_{timestamp}.log'
ensure_directory_exists(LOG_FILE_PATH)  # Ensure the directory exists before logging
logging.basicConfig(
    level=logging.INFO, # Minimum level of messages to log (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Send logs to a file
        # logging.StreamHandler() # Also send logs to console (optional, can be removed)
    ]
)
# Get a logger instance
logger = logging.getLogger(__name__)

# --- DataPreprocessor Class Definition ---
class DataPreprocessor:
    def __init__(self, index_column_name=None, numeric_columns=None, categorical_columns=None):
        self.index_column_name = index_column_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.fitted_params = {}

    def fit(self, df):
        logger.info("[Preprocessor - Fit] Starting to learn preprocessing parameters...")
        
        temp_df_for_type_detection = df.drop(columns=[self.index_column_name]) if self.index_column_name and self.index_column_name in df.columns else df
        
        if self.numeric_columns is None:
            self.numeric_columns = temp_df_for_type_detection.select_dtypes(include=np.number).columns.tolist()
        
        if self.categorical_columns is None:
            self.categorical_columns = temp_df_for_type_detection.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.numeric_columns:
            if col in df.columns:
                self.fitted_params[f'{col}_mean'] = df[col].mean()
        
        logger.info("[Preprocessor - Fit] Finished learning preprocessing parameters.")
        
    def transform(self, df):
        logger.info("[Preprocessor - Transform] Starting data transformations and cleaning...")
        processed_df = df.copy()

        for col in processed_df.select_dtypes(include=['object', 'category']).columns:
            processed_df[col] = processed_df[col].astype(str).str.strip()

        if self.index_column_name and self.index_column_name in processed_df.columns:
            initial_rows = len(processed_df)
            processed_df.dropna(subset=[self.index_column_name], inplace=True)
            if len(processed_df) < initial_rows:
                logger.info(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} rows due to nulls in index column '{self.index_column_name}'.")

        initial_rows = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        if len(processed_df) < initial_rows:
            logger.info(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} duplicate rows.")

        if self.numeric_columns:
            for col in self.numeric_columns:
                if col in processed_df.columns:
                    initial_nans = processed_df[col].isnull().sum()
                    if initial_nans > 0:
                        fill_value = self.fitted_params.get(f'{col}_mean', processed_df[col].mean()) 
                        # processed_df[col].fillna(fill_value, inplace=True)
                        processed_df[col] = processed_df[col].fillna(fill_value)

                        logger.info(f"[Preprocessor] Imputed {initial_nans} nulls in numeric column '{col}' with mean: {fill_value:.2f}.")

        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in processed_df.columns:
                    initial_rows_before_drop_cat_nan = len(processed_df)
                    processed_df.dropna(subset=[col], inplace=True)
                    if len(processed_df) < initial_rows_before_drop_cat_nan:
                        logger.info(f"[Preprocessor] Dropped {initial_rows_before_drop_cat_nan - len(processed_df)} rows due to nulls in categorical column '{col}'.")
        
        logger.info(f"[Preprocessor - Transform] Finished transformations and cleaning. Final rows: {len(processed_df)}")
        return processed_df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save_params(self, file_path):
        serializable_params = {k: v.item() if isinstance(v, np.generic) else v for k, v in self.fitted_params.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
        logger.info(f"[Preprocessor] Preprocessor parameters saved to: {file_path}")

    def load_params(self, file_path):
        with open(file_path, 'r') as f:
            self.fitted_params = json.load(f)
        logger.info(f"[Preprocessor] Preprocessor parameters loaded from: {file_path}")

# --- End of DataPreprocessor Class ---


# --- NaiveBayesClassifier Class Definition ---
class NaiveBayesClassifier:
    def __init__(self, target_column_name, smoothing_alpha=1, preprocessor=None):
        self.target_column_name = target_column_name
        self.prior_probabilities = {}
        self.likelihood_probabilities = {}
        self.feature_columns = []
        self.target_categories = []
        self.smoothing_alpha = smoothing_alpha
        self.preprocessor = preprocessor

    def _get_feature_columns(self, df, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []
        
        if self.target_column_name not in exclude_columns:
            exclude_columns.append(self.target_column_name)
            
        return [col for col in df.columns if col not in exclude_columns]

    def _calculate_prior_probabilities(self, df):
        prior_probs = df[self.target_column_name].value_counts(normalize=True).to_dict()
        return prior_probs

    def _calculate_likelihood_probabilities(self, df):
        likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for target_category in self.target_categories:
            subset_target = df[df[self.target_column_name] == target_category]
            target_count = len(subset_target)
            
            for feature_col in self.feature_columns:
                feature_unique_values = df[feature_col].unique()
                feature_counts = subset_target[feature_col].value_counts().to_dict()
                
                for feature_value in feature_unique_values:
                    count = feature_counts.get(feature_value, 0)
                    
                    likelihood = (count + self.smoothing_alpha) / (target_count + self.smoothing_alpha * len(feature_unique_values))
                    
                    likelihoods[feature_col][feature_value][target_category] = likelihood
        
        return {k: {vk: dict(vv) for vk, vv in v.items()} for k, v in likelihoods.items()}

    def train(self, train_df_path, excluded_columns_from_features=None):
        logger.info(f"\n[Classifier - Train] Starting Naive Bayes model training with file: {train_df_path}")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        train_df_raw = pd.read_csv(train_df_path)
        logger.info(f"[Classifier - Train] Loaded {len(train_df_raw)} raw training data rows.")

        processed_train_df = self.preprocessor.fit_transform(train_df_raw)
        
        if self.target_column_name not in processed_train_df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in DataFrame after preprocessing.")

        self.feature_columns = self._get_feature_columns(processed_train_df, excluded_columns_from_features)
        self.target_categories = processed_train_df[self.target_column_name].unique().tolist()
        logger.info(f"[Classifier - Train] Feature columns selected for model training: {self.feature_columns}")
        logger.info(f"[Classifier - Train] Target categories: {self.target_categories}")

        self.prior_probabilities = self._calculate_prior_probabilities(processed_train_df)
        logger.info("\n[Classifier - Train] Prior Probabilities calculated.")
        # logger.debug(f"Prior Probabilities: {self.prior_probabilities}") # Use debug for verbose output

        self.likelihood_probabilities = self._calculate_likelihood_probabilities(processed_train_df)
        logger.info(f"\n[Classifier - Train] Finished calculating likelihood probabilities for {len(self.likelihood_probabilities)} features.")
        # logger.debug(f"Likelihood Probabilities: {self.likelihood_probabilities}")

        logger.info("\n[Classifier - Train] Training process completed successfully.")

    def predict(self, predict_df_path, output_csv_path, excluded_columns_from_features=None, index_column_name=None):
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        logger.info(f"\n[Classifier - Predict] Starting prediction for file: {predict_df_path}")
        
        predict_df_raw = pd.read_csv(predict_df_path)
        logger.info(f"[Classifier - Predict] Loaded {len(predict_df_raw)} raw prediction data rows.")

        processed_predict_df = self.preprocessor.transform(predict_df_raw)
        
        original_index_col_values = None
        if index_column_name and index_column_name in processed_predict_df.columns:
            original_index_col_values = processed_predict_df[index_column_name]
            processed_predict_df = processed_predict_df.drop(columns=[index_column_name]) 

        predictions_list = []
        for index, row in processed_predict_df.iterrows():
            posterior_scores = {}
            for target_cat in self.target_categories:
                score = self.prior_probabilities.get(target_cat, 0)

                for feature_col in self.feature_columns:
                    feature_value = row[feature_col]

                    likelihood = self.likelihood_probabilities.get(feature_col, {}).get(feature_value, {}).get(target_cat, 
                                (self.smoothing_alpha / (len(self.target_categories) * self.smoothing_alpha)))
                    
                    score *= likelihood
                
                posterior_scores[target_cat] = score
            
            total_score = sum(posterior_scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in posterior_scores.items()}
            else: 
                normalized_scores = {k: 1 / len(self.target_categories) for k in self.target_categories} 

            predicted_class = max(normalized_scores, key=normalized_scores.get)
            predictions_list.append(predicted_class)
        
        output_df = pd.DataFrame({self.target_column_name: predictions_list})
        
        if original_index_col_values is not None:
            output_df.insert(0, index_column_name, original_index_col_values.reset_index(drop=True)) 
        
        output_df.to_csv(output_csv_path, index=False)
        logger.info(f"[Classifier - Predict] Prediction completed. Results saved to {output_csv_path}")
        return output_df

    def evaluate(self, predict_df_path, actual_results_path, excluded_columns_from_features=None, index_column_name=None):
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        logger.info(f"\n[Classifier - Evaluate] Starting performance evaluation...")

        temp_output_path = "temp_predictions_for_eval.csv"
        predicted_df = self.predict(predict_df_path, temp_output_path, excluded_columns_from_features, index_column_name)
        
        actual_df = pd.read_csv(actual_results_path)
        logger.info(f"[Classifier - Evaluate] Loaded {len(actual_df)} rows of actual results.")

        if index_column_name is None or index_column_name not in actual_df.columns:
            raise ValueError(f"Index column '{index_column_name}' not found in actual results file.")
        if self.target_column_name not in actual_df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in actual results file.")

        evaluation_df = pd.merge(predicted_df, actual_df[[index_column_name, self.target_column_name]], 
                                 on=index_column_name, 
                                 suffixes=('_predicted', '_actual'))
        
        correct_predictions = (evaluation_df[f'{self.target_column_name}_predicted'] == evaluation_df[f'{self.target_column_name}_actual']).sum()
        total_predictions = len(evaluation_df)
        
        accuracy = correct_predictions / total_predictions
        
        logger.info(f"\n--- [Classifier - Evaluate] Evaluation Results ---")
        logger.info(f"Total predictions: {total_predictions}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logger.info(f"[Classifier - Evaluate] Temporary file {temp_output_path} deleted.")
            
        logger.info("\n[Classifier - Evaluate] Performance evaluation completed.")
        return {"accuracy": accuracy}

# --- End of NaiveBayesClassifier Class ---


# --- Main Script Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Naive Bayes Model Script ---")

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
    logger.info("\n--- Step 1: Initializing DataPreprocessor ---")
    data_preprocessor = DataPreprocessor(
        index_column_name=INDEX_COL,
        numeric_columns=PREPROCESSOR_NUMERIC_COLS,
        categorical_columns=PREPROCESSOR_CATEGORICAL_COLS
    )

    # --- 2. Initialize the Classifier with the Preprocessor ---
    logger.info("\n--- Step 2: Initializing NaiveBayesClassifier ---")
    classifier = NaiveBayesClassifier(
        target_column_name=TARGET_COL,
        smoothing_alpha=1, # This value can be tuned
        preprocessor=data_preprocessor # Pass the preprocessor instance
    )

    # --- 3. Train the Model (Preprocessor will run internally on raw training data) ---
    logger.info("\n--- Step 3: Training the Model ---")
    try:
        classifier.train(train_df_path=TRAIN_RAW_CSV_PATH, 
                         excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB)
        
        # Optional: Save preprocessor parameters after training
        data_preprocessor.save_params(PREPROCESSOR_PARAMS_PATH)

    except Exception as e:
        logger.error(f"\nError during training: {e}", exc_info=True) # Log full exception info
        logger.error("Please ensure data files exist and column names in configurations are correct. Exiting.")
        exit()

    # --- 4. Predict (Preprocessor will run internally on raw prediction data) ---
    logger.info("\n--- Step 4: Making Predictions ---")
    try:
        predicted_df_output = classifier.predict(predict_df_path=TEST_RAW_CSV_PATH, 
                                                output_csv_path=PREDICTED_OUTPUT_CSV_PATH, 
                                                excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB, 
                                                index_column_name=INDEX_COL)
        logger.info(f"\nPredictions saved to {PREDICTED_OUTPUT_CSV_PATH}. Here are the first 5 rows (logged separately):")
        # Log the head of the DataFrame as string to avoid breaking log format
        logger.info(f"\n{predicted_df_output.head().to_string()}") 
    except Exception as e:
        logger.error(f"\nError during prediction: {e}", exc_info=True)
        logger.error("Exiting due to prediction error.")
        exit()

    # --- 5. Evaluate (Preprocessor will run internally) ---
    logger.info("\n--- Step 5: Evaluating Model Performance ---")
    try:
        evaluation_results = classifier.evaluate(predict_df_path=TEST_RAW_CSV_PATH, 
                                                actual_results_path=ACTUAL_RESULTS_CSV_PATH, 
                                                excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB,
                                                index_column_name=INDEX_COL)
        logger.info("\nFinal Evaluation Results:")
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}") # Log accuracy explicitly
    except Exception as e:
        logger.error(f"\nError during evaluation: {e}", exc_info=True)
        logger.error("Please ensure the actual results file contains the correct index and target columns. Exiting.")
        
    logger.info("\n--- Naive Bayes Model Script Finished ---")