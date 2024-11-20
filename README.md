# Anxiety-and-Depression-Model

# Overview
This project leverages Artificial Intelligence and Machine Learning (AIML) technologies to detect signs of anxiety and depression in textual data. By developing intelligent systems that analyze text, this project aims to provide insights into mental health challenges based on social media or other text data.

The implementation focuses on analyzing, preprocessing datasets, and using a supervised learning algorithm — Multinomial Naive Bayes. It demonstrates the effectiveness of text-based classification in detecting anxiety and depression.

# Purpose
**Mental Health Analysis:** Build a machine learning model to classify text as indicative of anxiety, depression, or neither, based on linguistic patterns.

# Technologies Used
1. **Programming Language:** Python
2. **Libraries:** Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn
3. **Development Environment:** Google Colab, Jupyter Notebook
4. **Algorithm:** Multinomial Naive Bayes

# Methodology
1. **Data Collection:** Gather labeled textual data, such as social media comments/posts, annotated with labels for anxiety, depression, or neutral sentiment.
2. **Data Preprocessing:**
   a) Clean text data (e.g., remove stop words, punctuation).
   b) Convert text to numerical features using TF-IDF vectorization.
3. **Model Training:** Train a Multinomial Naive Bayes Classifier on the processed data.
4. **Model Evaluation:** Evaluate the model using metrics such as accuracy and classification report.

# Dataset Information
1. **Text Samples:** Raw text data from various sources (e.g., social media).
2. **Labels:** Annotations indicating whether the text suggests anxiety, depression, or neither.
3. **Data Split:** 80% Training Data and 20% Testing Data

# Results
1. **Accuracy:** Achieved an accuracy of 89.59% on test data.
2. **Precision:**
   a) Class 0: 0.89
   b) Class 1: 1.00
3. **Recall:**
   a) Class 0: 1.00
   b) Class 1: 0.09
4. **F1 Score:**
   a) Class 0: 0.94
   b) Class 1: 0.16

# Key Insights
1. The Multinomial Naive Bayes classifier is efficient for text-based analysis, performing well with TF-IDF features.
2. Future iterations could explore more advanced models such as Support Vector Machines (SVM) or Neural Networks for improved accuracy.

# Conclusion
This project demonstrates the effectiveness of machine learning in identifying anxiety and depression using text data. By using supervised learning algorithms, it highlights the potential of AI-driven approaches in supporting mental health assessment and early detection.

# References
https://www.kaggle.com/datasets/sahasourav17/students-anxiety-and-depression-dataset
