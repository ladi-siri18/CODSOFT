# movie-genre-classification
This project aims to build a machine learning model that predicts the genre of a movie based on its plot summary or other textual information. The dataset contains various genres, and the model is trained to classify movies into one of the following categories:

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
Dataset
The dataset consists of movie plot summaries and their corresponding genres. Each entry includes:

Plot summary: A brief description of the movie plot.
Genre: The genre(s) assigned to the movie.
Model Training and Evaluation
The model uses natural language processing (NLP) techniques to process the text data and train the classifier. Below are the steps used for training and evaluation:

Text Preprocessing:
TF-IDF (Term Frequency-Inverse Document Frequency) is used to vectorize the plot summaries.
Classification Models:
Multiple classifiers are used for training the model:
Naive Bayes
Logistic Regression
Support Vector Machines (SVM)
Evaluation:
The classification performance is evaluated for each genre based on precision, recall, f1-score, and accuracy metrics.
Classification Reports
For each genre, the classification report includes:

Precision: The accuracy of the positive predictions.
Recall: The ability of the model to identify positive instances.
F1-score: The harmonic mean of precision and recall.
Accuracy: The overall correctness of the model.
Example of the classification report for Action genre:

txt
Copy code
--- Classification report for genre: Action ---
              precision    recall  f1-score   support
           0       0.88      0.82      0.85       709
           1       0.57      0.69      0.63       252

    accuracy                           0.78       961
   macro avg       0.73      0.75      0.74       961
weighted avg       0.80      0.78      0.79       961
Installation
To run this project, you'll need the following Python libraries:

scikit-learn (for machine learning models)
numpy (for numerical operations)
pandas (for data manipulation)
nltk or spaCy (for text processing)
matplotlib or seaborn (optional, for visualizations)
Install them using pip:

bash
Copy code
pip install scikit-learn numpy pandas nltk matplotlib seaborn
Usage
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/movie-genre-classification.git
Navigate to the project directory:

bash
Copy code
cd movie-genre-classification
Run the Python script to train the model:

bash
Copy code
python train_model.py
The model will output classification reports for each genre.

Conclusion
This project demonstrates the application of machine learning techniques for classifying movies based on their plot summaries. The model provides useful insights into genre classification and can be expanded for more genres or improved with additional preprocessing steps.
