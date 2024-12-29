# movie-genre-classification

This project implements a machine learning model to predict the genre of a movie based on its plot summary or other textual information. The model utilizes natural language processing (NLP) techniques and various classifiers to categorize movies into multiple genres.

#Genres Covered:
Action
Adventure
Animation
Comedy
Crime
Documentary
Drama
Family
Fantasy
Foreign
History
Horror
Music
Mystery
Romance
Science Fiction
TV Movie
Thriller
Unknown
War
Western
#Approach:
Text Preprocessing: The plot summaries are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
Classification Models: Models such as Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) are used for genre classification.
Evaluation: The model's performance is evaluated using precision, recall, F1-score, and accuracy for each genre.
Example Classification Report for Action Genre:
--- Classification report for genre: Action ---
              precision    recall  f1-score   support
           0       0.88      0.82      0.85       709
           1       0.57      0.69      0.63       252

    accuracy                           0.78       961
   macro avg       0.73      0.75      0.74       961
weighted avg       0.80      0.78      0.79       961
Installation
To run this project, you need to install the following dependencies:
pip install scikit-learn numpy pandas nltk
How to Run
Clone the repository:
git clone https://github.com/yourusername/movie-genre-classification.git
Navigate to the project folder:
cd movie-genre-classification
Train the model and evaluate:
python train_model.py
#Conclusion:
This project demonstrates how machine learning can be used for text classification, specifically for predicting movie genres. The model can be extended with additional genres or improved using other text processing techniques.



