from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import jsonify
from flask import Flask, request, jsonify
from flask import jsonify  # Import the jsonify function


# Define a function to encode input data using LabelEncoders
def encode_input_data(label_encoders, team1, team2, venue):
    encoded_data = {}
    print(team1,team2,venue)
    for column in ['Team1', 'Team2', 'Venue']:
        encoder = label_encoders[column]
        encoded_data[column] = encoder.transform([team1 if column == 'Team1' else (team2 if column == 'Team2' else venue)])[0]
    return encoded_data

# Define a function to predict the winner
def predict_winner(label_encoders, model, team1, team2, venue):
    # Encode the input data
    encoded_input = encode_input_data(label_encoders, team1, team2, venue)
    input_data = pd.DataFrame(encoded_input, index=[0])
    
    # Make prediction
    winner = model.predict(input_data)
    return winner[0]



app = Flask(__name__)

# Define routes and functions here
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Game=request.form['game']
    team1 = request.form['team1']
    team2 = request.form['team2']
    venue = request.form['venue']

    if Game=="Football - World Cup":
        model=joblib.load('data\\football_wc_model_lc.pkl')
        label_encoders=joblib.load('data\wc_le.pkl')
    elif Game=="Football - Champions League":
        model=joblib.load('data\\football_cl_model_lc.pkl')
        label_encoders=joblib.load('data\cl_le.pkl')
    elif Game=="Cricket - IPL":
        model=joblib.load('data\\cricket_ipl_model_lc.pkl')
        label_encoders=joblib.load('data\ipl_le.pkl')
    elif Game=="Cricket - World Cup":
        model=joblib.load('data\\cricket_ipl_model_lc.pkl')
        label_encoders=joblib.load('data\ipl_le.pkl')
    
    predicted_winner = predict_winner(label_encoders, model, team1, team2, venue)
    
    # Prepare the response data
    response = {'prediction': predicted_winner}

    # Return the response as JSON
    return jsonify(response)



if __name__ == '__main__':
    app.run()