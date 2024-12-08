from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Random Forest model (ensure the path is correct)
rf_model = joblib.load('random_forest_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve input values from the form
        population = int(request.form.get('population', 0))
        education = int(request.form.get('education', 0))
        unemployment = float(request.form.get('unemployment', 0.0))

        # Prepare the input data as a dictionary (make sure feature names match the model)
        new_data = {
            'population': [population],
            'education_pct_highschool': [education],  # Adjust according to model's expected feature
            'unemployment_rate': [unemployment]
        }

        # Convert the input data to a DataFrame
        new_data_df = pd.DataFrame(new_data)

        # If necessary, apply any scaling or transformations to new_data_df
        # Example (if you used StandardScaler or similar during training):
        # new_data_df = scaler.transform(new_data_df)

        # Make the prediction
        prediction = rf_model.predict(new_data_df)

        # Assuming the model predicts a Democratic vote share between 0 and 1
        result = "Party A" if prediction[0] > 0.5 else "Party B"

        # Display the result on the frontend
        return render_template('frontend.html', prediction=f"Predicted outcome: {result}")

    # Display the form when the request is GET
    return render_template('frontend.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
