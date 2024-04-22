from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
clf = joblib.load('model.joblib')
tfidf_vect = joblib.load('tfidf_vect.joblib')


app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from form
    new_text = request.form['new_text']
    
    # Vectorize the input text using the same TF-IDF vectorizer used for training
    new_text_tfidf = tfidf_vect.transform([new_text])

    # Make predictions using the trained logistic regression model
    prediction = clf.predict(new_text_tfidf)

    # Render prediction template with prediction result
    return render_template('prediction.html', prediction=prediction[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
