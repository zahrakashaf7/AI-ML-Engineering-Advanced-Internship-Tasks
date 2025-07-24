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




## Tools Used
- Python
- Hugging Face Transformers
- Datasets Library
- PyTorch
- Gradio
