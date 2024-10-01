# Task 1: POS Tagging and Named Entity Recognition (NER)

## Description
This task focuses on implementing and comparing different Part-of-Speech (POS) tagging and Named Entity Recognition (NER) models on a given dataset.

## Models Implemented
1. SpaCy
2. NLTK
3. Stanford NER

## Key Steps
1. Text Preprocessing
   - Data transformation
   - Sentence formation
   - Tag mapping for consistency across models

2. POS Tagging and NER for all three models
   - Implementation details for each model
   - Creation of separate columns for storing results

3. Model Evaluation
   - Accuracy calculation
   - Confusion matrix generation
   - Precision, Recall, and F1-score computation

## Results
- NLTK model achieved the highest accuracy of 96.25%
- SpaCy model achieved an accuracy of 95.80%
- Detailed confusion matrices and evaluation metrics are provided for each model

## Comparative Analysis
- NLTK showed slightly better performance across all metrics
- SpaCy demonstrated strengths in leveraging contextual information
- NLTK excelled in identifying certain POS tags like comparative adjectives and prepositions

## Observations on Errors
- Analysis of confusion matrices revealed insights into model strengths and weaknesses
- Discussion on the trade-offs between different approaches (e.g., rule-based vs. neural network-based)

## Tag Mapping Strategy
- Detailed explanation of the process for aligning tags from various datasets with model-specific tags
- Discussion on the importance of tag mapping for consistency and interoperability

## How to Use
1. Ensure you have SpaCy, NLTK, and Stanford NER installed
2. Run the preprocessing steps on your dataset
3. Execute each model's POS tagging and NER functions
4. Use the evaluation scripts to compare model performances

## Future Improvements
- Explore combining strengths of different models (e.g., SpaCy's contextual modeling with NLTK's rule-based approach)
- Investigate domain-specific fine-tuning for improved accuracy

## Contributors
- Drithi Davuluri (B21AI055)
- G Mukund (B21CS092)

## Date
February 20, 2024


# Task 2: Sentiment Analysis on Product Reviews

## Description
This task involves performing sentiment analysis on product reviews using various machine learning techniques. It includes both multi-class and binary classification approaches.

## Dataset
- 22 categories of product reviews
- Each review includes text content and a rating (1.0 to 5.0)
- Data is preprocessed and balanced using stratified sampling

## Classification Tasks
1. Multi-Class Classification (5 classes)
2. Binary Classification (Positive vs Negative)

## Models Implemented
1. Naive Bayes
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes
2. Decision Trees
   - With Entropy criterion
   - With Gini criterion
3. Random Forests
   - With 20, 50, and 100 trees

## Feature Extraction Techniques
1. Bag of Words (BoW)
2. TF-IDF

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrices
- ROC Curve (for Binary Classification)

## Key Findings
1. Multi-Class Classification
   - TF-IDF vectors of combined review + summary text provided the best performance
   - Random Forest with 100 trees achieved the highest accuracy of 62.54%
   - Gaussian Naive Bayes achieved 63.7% accuracy on TF-IDF features

2. Binary Classification
   - Multinomial Naive Bayes consistently outperformed Gaussian NB (>80% accuracy)
   - Random Forest achieved the best overall accuracy of 82-83% using TF-IDF features
   - All models showed better performance on the binary task compared to multi-class

## Hyperparameter Tuning
- Details on hyperparameters for each model and potential areas for improvement

## Error Analysis
- Discussion on model-specific limitations and general challenges in text classification

## How to Use
1. Ensure required libraries are installed (scikit-learn, NLTK, etc.)
2. Run data preparation and preprocessing scripts
3. Execute classification models for both multi-class and binary tasks
4. Analyze results using provided evaluation metrics

## Future Improvements
- Implement advanced NLP techniques for better semantic understanding
- Apply regularization and pruning to prevent overfitting
- Address class imbalance issues
- Explore deep learning models for potentially improved performance

## Contributors
- Drithi Davuluri (B21AI055)
- G Mukund (B21CS092)

## Date
February 20, 2024
