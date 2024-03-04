
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
app = Flask(__name__)




# Load the dataset
df = pd.read_csv('C:\\Users\\User\\Documents\\data con\\TMDB_cleaned_movie_dataset.csv')


# Preprocess the data
df['overview'] = df['overview'].str.lower()  # Convert to lowercase
# Add more preprocessing steps as needed

# Feature extraction
vectorizer = CountVectorizer(min_df=2, max_df=5000, stop_words="english")
vectorizer.fit(df.overview)
X = vectorizer.transform(df.overview)

genres_dummies = df['genres'].str.get_dummies(sep=',')
genres_dummies=genres_dummies.drop_duplicates()
df = pd.concat([df.drop(columns=['genres']), genres_dummies], axis=1)
df = df.drop_duplicates()

y=df[['Comedy','Crime','Action','Adventure','Animation','Documentary','Drama','Family','Fantasy',
      'History',	'Horror',	'Music','Mystery','Romance','Science Fiction','TV Movie',	'Thriller','War',	'Western']].fillna(0)

X_tr = X[:10000]
X_te = X[10000:]
y_tr = y[:10000]
y_te =y[10000:]

# Model building
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_tr, y_tr)

# Evaluation
y_pred = model.predict(X_te)

print('Accuracy:', accuracy_score(y_te, y_pred))





@app.route('/')
def index():
    return render_template('datacon.html')

@app.route('/classify', methods=['POST'])
def classify():
    input_text = request.form['text']
    input_vector = vectorizer.transform([input_text])
    print(f"Input text: {input_text}")  # Print the input text for debugging

    prediction = model.predict(input_vector)
    print(f"Prediction: {prediction}")  # Print the prediction for debugging
    return render_template('datacon.html', prediction=prediction.tolist(), input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)
