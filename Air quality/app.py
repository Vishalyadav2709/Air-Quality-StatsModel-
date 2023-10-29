from flask import Flask, render_template, request
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Sample air quality dataset (you should replace this with real data)
        data = pd.DataFrame({
            'Temperature': [22.5, 24.0, 25.5, 27.0, 28.5],
            'Humidity': [45, 42, 40, 38, 36],
            'Wind_Speed': [3.5, 3.2, 3.0, 2.8, 2.5],
            'Air_Quality': [30, 28, 25, 24, 23]
        })
        
        # Split the dataset into features and target
        X = data[['Temperature', 'Humidity', 'Wind_Speed']]
        y = data['Air_Quality']

        # Fit a linear regression model using statsmodels
        X = sm.add_constant(X)  # Add a constant (intercept) to the features
        model = sm.OLS(y, X).fit()

        # Get user input
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])

        # Make a prediction using the model
        prediction = model.predict([1, temperature, humidity, wind_speed])[0]

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
