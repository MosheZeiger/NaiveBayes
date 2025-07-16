import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import io
import logging

# Import your model logic
from .model import DataPreprocessor, NaiveBayesClassifier, setup_logging, ensure_directory_exists

# --- Configuration ---
# Set up logging for the API application
logger = setup_logging() # Initialize logger

# Paths for model and preprocessor parameters
# These paths are relative to the project root, assuming 'models' folder at the same level as 'app'
MODEL_DIR = "./models"
PREPROCESSOR_PARAMS_PATH = os.path.join(MODEL_DIR, 'preprocessor_params.json')
NAIVE_BAYES_MODEL_PATH = os.path.join(MODEL_DIR, 'naive_bayes_model.json')

# Model specific configurations (must match what was used during training)
TARGET_COL = 'Survived'
INDEX_COL = 'PassengerId'
PREPROCESSOR_NUMERIC_COLS = ['Age', 'SibSp', 'Parch', 'Fare']
PREPROCESSOR_CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked']
EXCLUDED_COLS_FROM_FEATURES_NB = [INDEX_COL] # Pass this to classifier methods if needed


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Naive Bayes Classifier API",
    description="API for training and predicting with a custom Naive Bayes Classifier for tabular data.",
    version="1.0.0"
)

# Global instances for preprocessor and classifier
# These will be loaded once when the application starts
data_preprocessor: Optional[DataPreprocessor] = None
classifier: Optional[NaiveBayesClassifier] = None

# --- Lifespan Events (Startup and Shutdown) ---
@app.on_event("startup")
async def startup_event():
    """
    Loads the preprocessor and trained Naive Bayes model on application startup.
    If models/params don't exist, it will log a warning.
    """
    global data_preprocessor, classifier
    
    logger.info("--- API Startup: Initializing Models ---")
    
    data_preprocessor = DataPreprocessor(
        index_column_name=INDEX_COL,
        numeric_columns=PREPROCESSOR_NUMERIC_COLS,
        categorical_columns=PREPROCESSOR_CATEGORICAL_COLS
    )
    
    classifier = NaiveBayesClassifier(
        target_column_name=TARGET_COL,
        preprocessor=data_preprocessor # Inject the preprocessor
    )

    try:
        # Attempt to load preprocessor params
        if os.path.exists(PREPROCESSOR_PARAMS_PATH):
            data_preprocessor.load_params(PREPROCESSOR_PARAMS_PATH)
            logger.info("Successfully loaded preprocessor parameters.")
        else:
            logger.warning(f"Preprocessor parameters file not found at {PREPROCESSOR_PARAMS_PATH}. Preprocessor will initialize without loaded parameters (might need re-fitting if used for prediction without prior training).")
            # If params are not found, numeric_columns might still be None in preprocessor
            # We explicitly set them during initialization, but if loaded from file, they might change

        # Attempt to load Naive Bayes model
        if os.path.exists(NAIVE_BAYES_MODEL_PATH):
            classifier.load_model(NAIVE_BAYES_MODEL_PATH, preprocessor=data_preprocessor)
            logger.info("Successfully loaded Naive Bayes model.")
        else:
            logger.warning(f"Naive Bayes model file not found at {NAIVE_BAYES_MODEL_PATH}. Model will be untrained. Use /train endpoint to train it.")

    except Exception as e:
        logger.error(f"Error during model/preprocessor loading: {e}", exc_info=True)
        logger.error("API will start, but model/preprocessor might not be ready for predictions until trained.")

@app.get("/")
async def read_root():
    """Root endpoint to check API status."""
    return {"message": "Naive Bayes Classifier API is running! Use /docs for API documentation."}

# --- API Endpoints ---

@app.post("/train/")
async def train_model(train_file: UploadFile = File(...)):
    """
    Trains the Naive Bayes Classifier model using the provided CSV file.
    The preprocessor will also be fitted during this process.
    """
    logger.info(f"Received training request for file: {train_file.filename}")
    
    try:
        # Read CSV file into pandas DataFrame
        contents = await train_file.read()
        train_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if classifier is None or data_preprocessor is None:
            raise HTTPException(status_code=500, detail="Model components not initialized. Check server logs.")

        # Train the model
        classifier.train(train_df_raw=train_df, 
                         excluded_columns_from_features=EXCLUDED_COLS_FROM_FEATURES_NB)
        
        # Save preprocessor params and trained model
        ensure_directory_exists(PREPROCESSOR_PARAMS_PATH) # Ensure directory for params
        data_preprocessor.save_params(PREPROCESSOR_PARAMS_PATH)
        ensure_directory_exists(NAIVE_BAYES_MODEL_PATH) # Ensure directory for model
        classifier.save_model(NAIVE_BAYES_MODEL_PATH)
        
        logger.info("Model training and saving completed successfully.")
        return {"message": "Model trained successfully and saved.", "model_status": "trained"}
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to train model: {e}")

@app.post("/predict/")
async def predict_data(predict_file: UploadFile = File(...)):
    """
    Makes predictions using the trained Naive Bayes Classifier on the provided CSV file.
    The model must be trained or loaded before making predictions.
    """
    logger.info(f"Received prediction request for file: {predict_file.filename}")

    if classifier is None or data_preprocessor is None:
        raise HTTPException(status_code=500, detail="Model components not initialized. Check server logs.")
    if not classifier.prior_probabilities or not classifier.likelihood_probabilities:
        raise HTTPException(status_code=400, detail="Model not trained or loaded. Please train the model first via /train endpoint or ensure model files exist on startup.")
    
    try:
        # Read CSV file into pandas DataFrame
        contents = await predict_file.read()
        predict_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Make predictions
        predictions_df = classifier.predict(predict_df_raw=predict_df, 
                                            index_column_name=INDEX_COL)
        
        # Convert predictions DataFrame to JSON serializable format
        # Use .to_dict(orient='records') for list of dictionaries, or .to_json()
        predictions_output = predictions_df.to_dict(orient='records')
        
        logger.info(f"Prediction completed for {len(predictions_output)} rows.")
        return {"predictions": predictions_output}
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {e}. Ensure input CSV matches expected format.")

@app.post("/evaluate/")
async def evaluate_model(
    predict_file: UploadFile = File(..., description="CSV file with data for prediction."),
    actual_results_file: UploadFile = File(..., description="CSV file with actual labels for evaluation.")
):
    """
    Evaluates the trained Naive Bayes Classifier's performance.
    Requires a prediction CSV and a CSV with actual results (containing index and target columns).
    """
    logger.info(f"Received evaluation request for prediction file: {predict_file.filename} and actuals file: {actual_results_file.filename}")

    if classifier is None or data_preprocessor is None:
        raise HTTPException(status_code=500, detail="Model components not initialized. Check server logs.")
    if not classifier.prior_probabilities or not classifier.likelihood_probabilities:
        raise HTTPException(status_code=400, detail="Model not trained or loaded. Please train the model first.")
    
    try:
        # Read prediction CSV
        predict_contents = await predict_file.read()
        predict_df = pd.read_csv(io.StringIO(predict_contents.decode('utf-8')))

        # Read actual results CSV
        actual_contents = await actual_results_file.read()
        actual_df = pd.read_csv(io.StringIO(actual_contents.decode('utf-8')))

        # Evaluate the model
        evaluation_results = classifier.evaluate(predict_df_raw=predict_df, 
                                                 actual_results_df_raw=actual_df, 
                                                 index_column_name=INDEX_COL)
        
        logger.info(f"Model evaluation completed. Accuracy: {evaluation_results['accuracy']:.4f}")
        return {"evaluation_results": evaluation_results}
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model: {e}. Ensure input CSVs match expected format.")