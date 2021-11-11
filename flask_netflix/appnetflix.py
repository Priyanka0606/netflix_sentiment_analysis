from flask import Flask, render_template, request
import joblib



# initialize the app
app = Flask(__name__)
vector = joblib.load('vectorizer_netflix.pkl')
model = joblib.load('netflix_99.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['post'])  
def predict():
    review = request.form.get("fname")

    print(review)
    tfidf = vector.transform([review])
    output = model.predict(tfidf)

    if output == 0:
        ans = 'Negative Review'
    else:
        ans = 'Positive Review'    
    
    return render_template('predict.html' , answer = f'This is a {ans}')  

# run the app
if __name__ == '__main__':
    app.run(debug=True)