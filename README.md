# AI-ML-Engineering-Advanced-Internship-Tasks

# Task 1: News Topic Classifier Using BERT

## Objective
The objective of this task is to build a News Topic Classifier using the BERT (`bert-base-uncased`) transformer model. The classifier predicts the topic category of a news headline, such as World, Sports, Business, or Sci/Tech, using the AG News dataset.

---

## Methodology / Approach

- **Dataset Used**: AG News Dataset (available via Hugging Face Datasets)
- **Model**: Pretrained BERT (`bert-base-uncased`) fine-tuned on the news headline data
- **Steps Followed**:
  1. Loaded and explored the AG News dataset
  2. Preprocessed the text using BERT tokenizer
  3. Fine-tuned the model using Hugging Face's `Trainer` API
  4. Evaluated model performance using accuracy and F1-score
  5. Built a Gradio web interface for live predictions

---

## Key Results / Observations

- ✅ The model was successfully trained and evaluated, achieving **75–85% accuracy** on a small training subset
- ✅ When tested through the Gradio interface, the model correctly classified real-world news headlines into the correct topics
- ✅ BERT performed well even with limited data, showing strong generalization for short-text classification
- 💡 This task demonstrates the effectiveness of transformer models like BERT for Natural Language Processing (NLP) tasks

---

## Tools Used
- Python
- Hugging Face Transformers
- Datasets Library
- PyTorch
- Gradio
