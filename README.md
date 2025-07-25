# AI-ML-Engineering-Advanced-Internship-Tasks
My tasks for the AI/ML Engineering Internship at DevelopersHub

# Task 1: News Topic Classifier Using BERT

## Objective
The objective of this task is to build a News Topic Classifier using the BERT (`bert-base-uncased`) transformer model. The classifier predicts the topic category of a news headline, such as World, Sports, Business, or Sci/Tech, using the AG News dataset.



## Methodology / Approach

- **Dataset Used**: AG News Dataset (available via Hugging Face Datasets)
- **Model**: Pretrained BERT (`bert-base-uncased`) fine-tuned on the news headline data
- **Steps Followed**:
  1. Loaded and explored the AG News dataset
  2. Preprocessed the text using BERT tokenizer
  3. Fine-tuned the model using Hugging Face's `Trainer` API
  4. Evaluated model performance using accuracy and F1-score
  5. Built a Gradio web interface for live predictions



## Key Results / Observations

- The model was successfully trained and evaluated, achieving **75–85% accuracy** on a small training subset
- When tested through the Gradio interface, the model correctly classified real-world news headlines into the correct topics
- BERT performed well even with limited data, showing strong generalization for short-text classification
- This task demonstrates the effectiveness of transformer models like BERT for Natural Language Processing (NLP) tasks

  -----

# Task 2: Telco Customer Churn Prediction using End-to-End ML Pipeline

## Objective of the Task
The goal of this task was to build a complete end-to-end machine learning pipeline that predicts whether a telecom customer is likely to churn. The pipeline needed to automate preprocessing, model training, hyperparameter tuning, and evaluation — making it production-ready and reusable.


## Methodology / Approach

1. **Dataset Used**:  
   - Telco Customer Churn Dataset (from Kaggle)

2. **Steps Followed**:
   - Loaded and explored the dataset using `pandas`
   - Handled missing values and encoded categorical variables
   - Built a `Pipeline` using Scikit-learn to chain preprocessing and modeling steps
   - Used a `RandomForestClassifier` as the base model
   - Applied `GridSearchCV` to tune hyperparameters:
     - `n_estimators`: Number of trees
     - `max_depth`: Maximum depth of each tree
   - Evaluated performance using:
     - Accuracy
     - Classification Report (Precision, Recall, F1-Score)
     - Confusion Matrix
   - Exported the final trained model using `joblib`



## Key Results or Observations

- **Best Parameters**:  
  `{'model__max_depth': 10, 'model__n_estimators': 100}`

- **Best Cross-Validation Accuracy**:  
  `~79.5%`

- **Test Accuracy**:  
  `~79.7%`

- **Confusion Matrix**:
  [[946 90]
  [196 177]]

-------

# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

## Objective of the Task

The objective of this task is to develop a **Multimodal Machine Learning model** that predicts the **price of houses** using a combination of:
- **Tabular data** (e.g., number of bedrooms, bathrooms, square footage)
- **Image data** (photos of the houses)

This real-world scenario showcases how structured and unstructured data can be combined to improve predictive performance.



## Methodology / Approach

1. **Dataset Loading**
   - Loaded the tabular dataset (`socal2.csv`) containing housing attributes and prices.
   - Loaded corresponding house images from the `socal_pics` directory using `image_id`.

2. **Tabular Data Preprocessing**
   - Selected relevant features: `bed`, `bath`, `sqft`.
   - Normalized the target variable `price` by scaling down to thousands.

3. **Image Preprocessing**
   - Loaded and resized all house images to 128x128.
   - Converted images to NumPy arrays and normalized pixel values.

4. **Image Feature Extraction**
   - Used a pre-trained **MobileNetV2** model with `pooling='avg'` to extract deep image features.
   - Each image was represented as a fixed-length vector.

5. **Combining Features**
   - Concatenated tabular and image features to create a single feature set for each sample.

6. **Model Training**
   - Trained a **Random Forest Regressor** on the combined feature set.
   - Used an 80/20 train-test split.

7. **Model Evaluation**
   - Evaluated model using:
     - **R² Score**
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
   - Plotted predicted vs. actual prices.


## Key Results or Observations

- **R² Score**: `0.4435`  
- **Mean Absolute Error (MAE)**: `209,714`  
- **Root Mean Squared Error (RMSE)**: `286,387`  

🔹 The model moderately captured the relationship between the features and house prices.  
🔹 Multimodal learning (image + tabular) added value to the regression task.  
🔹 Further improvements can be made by:
  - Fine-tuning CNN layers
  - Using more expressive models like XGBoost or deep MLPs
  - Improving image quality and alignment

---



