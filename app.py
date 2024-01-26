import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("Student_mark_predictor_model.pkl")

# Initialize an empty DataFrame
data = pd.DataFrame(columns=["Study Hours", "Predicted Output"])

@app.route('/')
def home():
    return render_template('index.html', p_text=" ")

@app.route('/pred', methods=['POST'])
def pred():
    global data
    
    try:
        # Get study hours from the form
        study_hours = float(request.form['study_hours'])
        
        # Check if study hours are within a reasonable range
        if study_hours < 0 or study_hours > 24:
            raise ValueError("Invalid input. Study hours should be between 0 and 24.")
        if  study_hours == 24 or study_hours > 12:
            raise ValueError("You will Sick!! you should sleep 6-8 hours in a day")
        if study_hours==0:
            raise ValueError("Sorry to say but you should study..otherwise you will fail!! Start studying from today")
        # Make prediction using the model
        predicted_output = model.predict([[study_hours]])[0].round(2)
        
        # Update the DataFrame with the current prediction
        data = data.append({"Study Hours": study_hours, "Predicted Output": predicted_output}, ignore_index=True)
        data.to_csv('data_from_app.csv', index=False)
        
        # Render the template with the prediction result
        return render_template('index.html', p_text=f'You can obtain {predicted_output}% marks')

    except Exception as e:
        # Handle exceptions and display an error message
        return render_template('index.html', p_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
