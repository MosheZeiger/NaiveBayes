import pandas as pd
import numpy as np
import json
import os
import logging
from collections import defaultdict
from datetime import datetime

# --- Utility Function: ensure_directory_exists ---
def ensure_directory_exists(path):
    """
    Ensures that the directory for a given path exists.
    If the path includes a filename, it ensures the parent directory exists.
    """
    if os.path.basename(path) and '.' in os.path.basename(path):
        directory_path = os.path.dirname(path)
    else:
        directory_path = path

    if not directory_path:
        directory_path = '.' 

    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as e:
            print(f"ERROR: Could not create directory {directory_path}: {e}")
            raise


# --- Logger Configuration (Moved to be more flexible, but kept for clarity here) ---
# In a real app, you might configure logging once at the application's entry point.
# For this example, we'll ensure it's configured when the module is imported/used.
# We'll use a global logger setup that can be called once.
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-4]
    LOG_FILE_PATH = f'./log/model_run_{timestamp}.log'
    ensure_directory_exists(LOG_FILE_PATH) 
    
    # Check if a root handler already exists to prevent duplicate logs
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE_PATH),
                logging.StreamHandler() # Keep stream handler for console visibility during development
            ]
        )
    return logging.getLogger(__name__)

logger = setup_logging() # Initialize logger when module is imported

# --- DataPreprocessor Class Definition ---
class DataPreprocessor:
    def __init__(self, index_column_name=None, numeric_columns=None, categorical_columns=None):
        self.index_column_name = index_column_name
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.fitted_params = {}

    def fit(self, df):
        logger.info("[Preprocessor - Fit] Starting to learn preprocessing parameters...")
        
        # Determine columns if not explicitly provided
        temp_df_for_type_detection = df.drop(columns=[self.index_column_name]) if self.index_column_name and self.index_column_name in df.columns else df
        
        if self.numeric_columns is None:
            self.numeric_columns = temp_df_for_type_detection.select_dtypes(include=np.number).columns.tolist()
            logger.info(f"[Preprocessor] Auto-detected numeric columns: {self.numeric_columns}")
        
        if self.categorical_columns is None:
            self.categorical_columns = temp_df_for_type_detection.select_dtypes(include=['object', 'category']).columns.tolist()
            logger.info(f"[Preprocessor] Auto-detected categorical columns: {self.categorical_columns}")
        
        # Calculate means for numeric columns
        for col in self.numeric_columns:
            if col in df.columns:
                self.fitted_params[f'{col}_mean'] = df[col].mean()
                logger.debug(f"Calculated mean for {col}: {self.fitted_params[f'{col}_mean']:.2f}")
            else:
                logger.warning(f"[Preprocessor - Fit] Numeric column '{col}' not found in DataFrame. Skipping mean calculation.")
        
        logger.info("[Preprocessor - Fit] Finished learning preprocessing parameters.")
        
    def transform(self, df):
        logger.info("[Preprocessor - Transform] Starting data transformations and cleaning...")
        processed_df = df.copy()

        # Strip whitespace from categorical/object columns
        for col in processed_df.select_dtypes(include=['object', 'category']).columns:
            # Check if column exists and has string data
            if col in processed_df.columns and processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].astype(str).str.strip()
                logger.debug(f"Stripped whitespace from categorical column '{col}'.")


        # Handle index column nulls
        if self.index_column_name and self.index_column_name in processed_df.columns:
            initial_rows = len(processed_df)
            processed_df.dropna(subset=[self.index_column_name], inplace=True)
            if len(processed_df) < initial_rows:
                logger.info(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} rows due to nulls in index column '{self.index_column_name}'.")
        else:
            if self.index_column_name: # Only log if index_column_name was actually provided but not found
                logger.warning(f"[Preprocessor] Index column '{self.index_column_name}' not found in DataFrame for null handling.")


        # Drop duplicate rows
        initial_rows = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        if len(processed_df) < initial_rows:
            logger.info(f"[Preprocessor] Dropped {initial_rows - len(processed_df)} duplicate rows.")

        # Impute numeric columns
        if self.numeric_columns:
            for col in self.numeric_columns:
                if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
                    initial_nans = processed_df[col].isnull().sum()
                    if initial_nans > 0:
                        fill_value = self.fitted_params.get(f'{col}_mean')
                        if fill_value is None: # Fallback if params not loaded or column not in fit
                            fill_value = processed_df[col].mean()
                            logger.warning(f"[Preprocessor] Mean for '{col}' not found in fitted_params. Calculating from current data: {fill_value:.2f}.")
                        
                        processed_df[col] = processed_df[col].fillna(fill_value)
                        logger.info(f"[Preprocessor] Imputed {initial_nans} nulls in numeric column '{col}' with mean: {fill_value:.2f}.")
                else:
                    logger.warning(f"[Preprocessor - Transform] Numeric column '{col}' not found or not numeric in DataFrame. Skipping imputation.")


        # Handle categorical column nulls (dropping rows with nulls)
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in processed_df.columns:
                    initial_rows_before_drop_cat_nan = len(processed_df)
                    processed_df.dropna(subset=[col], inplace=True)
                    if len(processed_df) < initial_rows_before_drop_cat_nan:
                        logger.info(f"[Preprocessor] Dropped {initial_rows_before_drop_cat_nan - len(processed_df)} rows due to nulls in categorical column '{col}'.")
                else:
                    logger.warning(f"[Preprocessor - Transform] Categorical column '{col}' not found in DataFrame. Skipping null handling.")

        logger.info(f"[Preprocessor - Transform] Finished transformations and cleaning. Final rows: {len(processed_df)}")
        return processed_df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save_params(self, file_path):
        ensure_directory_exists(file_path) # Ensure directory exists before saving
        serializable_params = {k: v.item() if isinstance(v, np.generic) else v for k, v in self.fitted_params.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)
        logger.info(f"[Preprocessor] Preprocessor parameters saved to: {file_path}")

    def load_params(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor parameters file not found: {file_path}")
        with open(file_path, 'r') as f:
            self.fitted_params = json.load(f)
        logger.info(f"[Preprocessor] Preprocessor parameters loaded from: {file_path}")
        # Re-set column lists based on loaded params if they were None
        if self.numeric_columns is None:
            self.numeric_columns = sorted([k.replace('_mean', '') for k in self.fitted_params.keys() if '_mean' in k])
            logger.info(f"[Preprocessor] Numeric columns set from loaded params: {self.numeric_columns}")


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

    def _get_feature_columns(self, df, excluded_columns=None):
        if excluded_columns is None:
            excluded_columns = []
        
        # Ensure target column is always excluded
        if self.target_column_name not in excluded_columns:
            excluded_columns.append(self.target_column_name)
            
        return [col for col in df.columns if col not in excluded_columns]

    def _calculate_prior_probabilities(self, df):
        if self.target_column_name not in df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in DataFrame for prior probability calculation.")
        prior_probs = df[self.target_column_name].value_counts(normalize=True).to_dict()
        return prior_probs

    def _calculate_likelihood_probabilities(self, df):
        likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Ensure target categories and feature columns are set before calculation
        if not self.target_categories:
            raise RuntimeError("Target categories must be set before calculating likelihood probabilities.")
        if not self.feature_columns:
            raise RuntimeError("Feature columns must be set before calculating likelihood probabilities.")

        for target_category in self.target_categories:
            subset_target = df[df[self.target_column_name] == target_category]
            target_count = len(subset_target)
            
            for feature_col in self.feature_columns:
                # Get all unique values for the feature *across the entire training data*
                # This is important for smoothing to work correctly for unseen feature values
                feature_unique_values_overall = df[feature_col].unique() 
                feature_counts = subset_target[feature_col].value_counts().to_dict()
                
                for feature_value in feature_unique_values_overall: # Iterate over ALL possible values
                    count = feature_counts.get(feature_value, 0)
                    
                    likelihood = (count + self.smoothing_alpha) / (target_count + self.smoothing_alpha * len(feature_unique_values_overall))
                    
                    likelihoods[feature_col][feature_value][target_category] = likelihood
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return {k: {vk: dict(vv) for vk, vv in v.items()} for k, v in likelihoods.items()}

    def train(self, train_df_raw, excluded_columns_from_features=None):
        logger.info(f"\n[Classifier - Train] Starting Naive Bayes model training with {len(train_df_raw)} raw training data rows.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        processed_train_df = self.preprocessor.fit_transform(train_df_raw)
        
        if self.target_column_name not in processed_train_df.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in DataFrame after preprocessing.")
        
        # Ensure target column is of string/category type
        processed_train_df[self.target_column_name] = processed_train_df[self.target_column_name].astype(str)

        self.feature_columns = self._get_feature_columns(processed_train_df, excluded_columns_from_features)
        self.target_categories = processed_train_df[self.target_column_name].unique().tolist()
        logger.info(f"[Classifier - Train] Feature columns selected for model training: {self.feature_columns}")
        logger.info(f"[Classifier - Train] Target categories: {self.target_categories}")

        self.prior_probabilities = self._calculate_prior_probabilities(processed_train_df)
        logger.info("[Classifier - Train] Prior Probabilities calculated.")
        logger.debug(f"Prior Probabilities: {self.prior_probabilities}")

        self.likelihood_probabilities = self._calculate_likelihood_probabilities(processed_train_df)
        logger.info(f"[Classifier - Train] Finished calculating likelihood probabilities for {len(self.likelihood_probabilities)} features.")
        logger.debug(f"Likelihood Probabilities: {self.likelihood_probabilities}")

        logger.info("[Classifier - Train] Training process completed successfully.")

    def predict(self, predict_df_raw, index_column_name=None):
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first or load a trained model.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        logger.info(f"[Classifier - Predict] Starting prediction for {len(predict_df_raw)} raw prediction data rows.")
        
        processed_predict_df = self.preprocessor.transform(predict_df_raw)
        
        original_index_col_values = None
        if index_column_name and index_column_name in processed_predict_df.columns:
            original_index_col_values = processed_predict_df[index_column_name].copy() # Use .copy() to avoid SettingWithCopyWarning
            # Remove index column from features used for prediction
            processed_predict_df = processed_predict_df.drop(columns=[index_column_name]) 
        
        predictions_list = []
        for index, row in processed_predict_df.iterrows():
            posterior_scores = {}
            for target_cat in self.target_categories:
                # Initialize score with prior probability
                score = self.prior_probabilities.get(target_cat, 0)
                if score == 0: # If prior is 0 for some reason (e.g., category not in training data), assign a small value
                    score = 1e-10 # Smallest positive float for logs/multiplications

                for feature_col in self.feature_columns:
                    feature_value = row.get(feature_col) # Use .get() for robustness in case column is missing

                    # Retrieve likelihood; if feature_value not seen for this feature_col, apply smoothing formula
                    # Need to know the total unique values of the feature from TRAINING data for proper smoothing
                    # This implies self.preprocessor should track unique categorical values if we rely on it
                    # For now, we'll try to find it in the stored likelihoods.
                    # If feature_value is completely unseen for this feature_col, apply baseline smoothed likelihood
                    
                    # Fallback for unseen feature_value during prediction:
                    # The number of unique values for a feature (len(feature_unique_values)) used in the denominator
                    # should come from the *training* data. We need to store this during training.
                    
                    # For a robust solution, during `train`, you'd store all unique values per feature.
                    # For now, a simplified fallback:
                    
                    # Try to get likelihood from stored model
                    likelihood_val = self.likelihood_probabilities.get(feature_col, {}).get(feature_value, {}).get(target_cat)
                    
                    if likelihood_val is None:
                        # Fallback for entirely unseen feature_value within a feature_col or missing feature_col
                        # This fallback is crucial for robust prediction
                        # A better approach involves storing `len(feature_unique_values)` during training
                        # For now, let's estimate a small default for unseen feature values
                        # This assumes we have at least 1 unique feature value from training.
                        
                        # Find the feature_col's total unique values from the stored likelihoods keys
                        # This is a bit of a hack, better to store this explicitly during training.
                        
                        # Estimate total unique feature values for this column based on what was learned
                        # This might be inaccurate if some unique values had 0 count for all target_cats
                        # A more robust solution involves storing a mapping of {feature_col: set_of_all_unique_values_from_training}
                        # when `train` is called.
                        estimated_total_unique_feature_values = len(self.likelihood_probabilities.get(feature_col, {}))
                        if estimated_total_unique_feature_values == 0:
                            # If no unique values were learned for this feature_col, assume 1 to avoid division by zero
                            estimated_total_unique_feature_values = 1 
                            
                        likelihood_val = (self.smoothing_alpha) / (subset_target.shape[0] + self.smoothing_alpha * estimated_total_unique_feature_values)
                        
                        logger.debug(f"Unseen feature value '{feature_value}' for column '{feature_col}' or missing feature. Applying smoothed default likelihood: {likelihood_val:.4f}")
                    
                    score *= likelihood_val
                
                posterior_scores[target_cat] = score
            
            total_score = sum(posterior_scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in posterior_scores.items()}
            else: 
                # Fallback if all scores are zero (e.g., due to log(0) issues or very small numbers)
                normalized_scores = {k: 1 / len(self.target_categories) for k in self.target_categories} 
                logger.warning(f"All posterior scores were zero for a row. Assigning uniform probabilities. Scores: {posterior_scores}")

            predicted_class = max(normalized_scores, key=normalized_scores.get)
            predictions_list.append(predicted_class)
        
        output_df = pd.DataFrame({self.target_column_name: predictions_list})
        
        if original_index_col_values is not None:
            # Re-align index values if any rows were dropped during preprocessing
            if len(original_index_col_values) != len(output_df):
                logger.warning(f"Mismatch in row count between original index and predicted output ({len(original_index_col_values)} vs {len(output_df)}). This might happen if rows were dropped during preprocessing. Predictions will be returned without original index for clarity.")
                # If rows were dropped, the original_index_col_values won't directly map.
                # In a real scenario, you'd want to return the original row index from processed_predict_df
                # processed_predict_df has already handled drops, so its index might be the correct one
                if index_column_name in predict_df_raw.columns:
                     # Attempt to align by index if index_column_name was originally present in raw data
                     # and if processed_predict_df still retains its original index from raw data
                     # This is tricky if dropna/drop_duplicates change the index non-sequentially.
                     # A safer way would be to carry a unique ID through processing if not the index_column_name.
                     # For simplicity, if mismatch, we'll just not add the index.
                    output_df.insert(0, index_column_name, processed_predict_df_raw[index_column_name].reset_index(drop=True))
                else:
                    logger.warning("Cannot re-attach original index due to row count mismatch after preprocessing and missing original index column in raw data.")
            else:
                 output_df.insert(0, index_column_name, original_index_col_values.reset_index(drop=True)) 

        logger.info(f"[Classifier - Predict] Prediction completed. Generated {len(output_df)} predictions.")
        return output_df

    def evaluate(self, predict_df_raw, actual_results_df_raw, index_column_name=None):
        if not self.prior_probabilities or not self.likelihood_probabilities:
            raise RuntimeError("Model not trained. Please run the train() method first.")
        
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor instance must be provided to NaiveBayesClassifier.")

        logger.info(f"[Classifier - Evaluate] Starting performance evaluation...")

        # Predict based on the raw prediction DataFrame
        predicted_df_output = self.predict(predict_df_raw, index_column_name=index_column_name)
        
        # Process actual results DataFrame
        # IMPORTANT: If actual_results_df_raw also needs preprocessing (e.g., dropping nulls),
        # it should go through the preprocessor. For simplicity here, assuming it's clean enough
        # for merge, or only needs a subset of cols.
        actual_df_processed = actual_results_df_raw.copy()
        
        if index_column_name is None or index_column_name not in actual_df_processed.columns:
            raise ValueError(f"Index column '{index_column_name}' not found in actual results DataFrame for evaluation.")
        if self.target_column_name not in actual_df_processed.columns:
            raise ValueError(f"Target column '{self.target_column_name}' not found in actual results DataFrame for evaluation.")

        # Ensure target column in actuals is string/category for consistent comparison
        actual_df_processed[self.target_column_name] = actual_df_processed[self.target_column_name].astype(str)

        # Merge predicted and actual results on the index column
        evaluation_df = pd.merge(predicted_df_output, actual_df_processed[[index_column_name, self.target_column_name]], 
                                 on=index_column_name, 
                                 suffixes=('_predicted', '_actual'))
        
        correct_predictions = (evaluation_df[f'{self.target_column_name}_predicted'] == evaluation_df[f'{self.target_column_name}_actual']).sum()
        total_predictions = len(evaluation_df)
        
        accuracy = correct_predictions / total_predictions
        
        logger.info(f"\n--- [Classifier - Evaluate] Evaluation Results ---")
        logger.info(f"Total evaluated predictions: {total_predictions}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Accuracy: {accuracy:.4f}")
            
        logger.info("[Classifier - Evaluate] Performance evaluation completed.")
        return {"accuracy": accuracy}

    def save_model(self, model_path):
        """Saves the trained model (prior and likelihood probabilities) to a JSON file."""
        model_data = {
            "prior_probabilities": self.prior_probabilities,
            "likelihood_probabilities": self.likelihood_probabilities,
            "feature_columns": self.feature_columns,
            "target_categories": self.target_categories,
            "smoothing_alpha": self.smoothing_alpha,
            "target_column_name": self.target_column_name
        }
        ensure_directory_exists(model_path) # Ensure directory exists before saving
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        logger.info(f"[Classifier] Model saved to: {model_path}")

    def load_model(self, model_path, preprocessor):
        """Loads a trained model from a JSON file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.prior_probabilities = model_data.get("prior_probabilities", {})
        self.likelihood_probabilities = model_data.get("likelihood_probabilities", {})
        self.feature_columns = model_data.get("feature_columns", [])
        self.target_categories = model_data.get("target_categories", [])
        self.smoothing_alpha = model_data.get("smoothing_alpha", 1) # Default to 1 if not present
        self.target_column_name = model_data.get("target_column_name", self.target_column_name)
        
        # Ensure preprocessor is set if loading a model, as it's crucial for prediction
        self.preprocessor = preprocessor 
        
        logger.info(f"[Classifier] Model loaded from: {model_path}")
        logger.debug(f"Loaded feature columns: {self.feature_columns}")
        logger.debug(f"Loaded target categories: {self.target_categories}")

# --- End of NaiveBayesClassifier Class ---

# --- Main Script Execution (if run directly) ---
if __name__ == "__main__":
    logger.info("--- Starting Naive Bayes Model Script (Local Run) ---")

    # --- 0. Configurations and Paths ---
    # Ensure these files exist in your script's directory or provide full paths
    TRAIN_RAW_CSV_PATH = 'train.csv' 
    TEST_RAW_CSV_PATH = 'test.csv'   
    ACTUAL_RESULTS_CSV_PATH = 'gender_submission.csv' 

    PREDICTED_OUTPUT_CSV_PATH = 'output/titanic_predictions.csv' # Changed path to 'output'
    PREPROCESSOR_PARAMS_PATH = 'models/preprocessor_params.json' # Changed path to 'models'
    MODEL_PATH = 'models/naive_bayes_model.json' # New path for model parameters

    TARGET_COL = 'Survived' 
    INDEX_COL = 'PassengerId' 

    PREPROCESSOR_NUMERIC_COLS = ['Age', 'SibSp', 'Parch', 'Fare']
    PREPROCESSOR_CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked'] 

    EXCLUDED_COLS_FROM_FEATURES_NB = [INDEX_COL] 

    # Ensure output and model directories exist
    ensure_directory_exists(PREDICTED_OUTPUT_CSV_PATH)
    ensure_directory_exists(PREPROCESSOR_PARAMS_PATH)
    ensure_directory_exists(MODEL_PATH)


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
        smoothing_alpha=1, 
        preprocessor=data_preprocessor 
    )

    # --- 3. Train the Model ---
    logger.info("\n--- Step 3: Training the Model ---")
    try:
        # Load raw dataframes here for the training process
        train_df_raw = pd.read_csv(TRAIN_RAW_CSV_PATH)
        classifier.train(train_df_raw=train_df_raw, 
                         excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB)
        
        # Save preprocessor parameters and trained model
        data_preprocessor.save_params(PREPROCESSOR_PARAMS_PATH)
        classifier.save_model(MODEL_PATH)

    except Exception as e:
        logger.error(f"\nError during training: {e}", exc_info=True)
        logger.error("Please ensure data files exist and column names in configurations are correct. Exiting.")
        exit()

    # --- 4. Predict ---
    logger.info("\n--- Step 4: Making Predictions ---")
    try:
        test_df_raw = pd.read_csv(TEST_RAW_CSV_PATH)
        predicted_df_output = classifier.predict(predict_df_raw=test_df_raw, 
                                                 index_column_name=INDEX_COL)
        
        # Save predictions to CSV
        ensure_directory_exists(PREDICTED_OUTPUT_CSV_PATH) # Ensure output directory exists
        predicted_df_output.to_csv(PREDICTED_OUTPUT_CSV_PATH, index=False)

        logger.info(f"\nPredictions saved to {PREDICTED_OUTPUT_CSV_PATH}. Here are the first 5 rows (logged separately):")
        logger.info(f"\n{predicted_df_output.head().to_string()}") 
    except Exception as e:
        logger.error(f"\nError during prediction: {e}", exc_info=True)
        logger.error("Exiting due to prediction error.")
        exit()

    # --- 5. Evaluate ---
    logger.info("\n--- Step 5: Evaluating Model Performance ---")
    try:
        actual_results_df = pd.read_csv(ACTUAL_RESULTS_CSV_PATH) # Load actual results for evaluation
        evaluation_results = classifier.evaluate(predict_df_raw=test_df_raw, # Pass the raw test data again
                                                 actual_results_df_raw=actual_results_df, # Pass the raw actuals
                                                 index_column_name=INDEX_COL)
        logger.info("\nFinal Evaluation Results:")
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"\nError during evaluation: {e}", exc_info=True)
        logger.error("Please ensure the actual results file contains the correct index and target columns. Exiting.")
        
    logger.info("\n--- Naive Bayes Model Script Finished ---")