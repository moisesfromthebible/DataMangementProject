from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature order (ensure the path is correct)
model_data = joblib.load('random_forest_model.pkl')
rf_model = model_data['model']
feature_order = model_data['feature_order']

# Ensure the features are correctly prepared for the input
def prepare_input_data(form_data):
    # Prepare the input dictionary
    data = {
        'avg_pop_estimate': [form_data['population']],
        'avg_unemployment_rate': [form_data['unemployment']],
        'avg_bachelor_or_higher': [form_data['education']],
        'avg_high_school_diploma_only': [form_data['education']],
        'avg_less_than_high_school_diploma': [form_data['education']],
        'avg_some_college_1_3_years': [form_data['education']],
        'avg_some_college_or_associate_degree': [form_data['education']],
        'avg_four_years_college_or_higher': [form_data['education']]
    }

    # Create a DataFrame and reorder columns to match training order
    input_df = pd.DataFrame(data)
    input_df = input_df[feature_order]
    return input_df

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve input values from the form
        try:
            population = float(request.form.get('population', 0))
            education = float(request.form.get('education', 0))
            unemployment = float(request.form.get('unemployment', 0.0))

            # Prepare the input data (same as during training)
            input_data = prepare_input_data({
                'population': population,
                'education': education,
                'unemployment': unemployment
            })

            # Make the prediction
            prediction = rf_model.predict(input_data)

            # Interpret the prediction (Democratic vote share or party outcome)
            result = "Democrat" if prediction[0] > 0.5 else "Republican"

            # Display the result on the frontend
            return render_template('frontend.html', prediction=f"Predicted outcome: {result}")
        except Exception as e:
            return render_template('frontend.html', prediction=f"Error: {str(e)}")

    # Display the form when the request is GET
    return render_template('frontend.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
