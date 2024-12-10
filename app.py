from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature order
model_data = joblib.load('random_forest_model.pkl')
rf_model = model_data['model']
feature_order = model_data['feature_order']

def prepare_input_data(form_data):
    # Create a dictionary of all required features with user input
    data = {
        'population_est': [form_data['population']],
        'edu_attainment_rate': [form_data['education']],
        'unemp_rate': [form_data['unemployment']]
    }

    # Create a DataFrame and reorder columns to match training order
    input_df = pd.DataFrame(data)
    input_df = input_df[feature_order]
    return input_df

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            population = float(request.form.get('population', 0))
            education = float(request.form.get('education', 0))
            unemployment = float(request.form.get('unemployment', 0))

            # Prepare the input data
            input_data = {
                'population': population,
                'education': education,
                'unemployment': unemployment
            }

            input_df = prepare_input_data(input_data)

            # Make the prediction
            prediction = rf_model.predict(input_df)
            predicted_party = "Democrat" if prediction[0] == 0 else "Republican"

            # Get prediction probabilities
            prediction_prob = rf_model.predict_proba(input_df)
            prob_df = pd.DataFrame(prediction_prob, columns=['DEMOCRAT', 'REPUBLICAN'])

            # Render template with results
            return render_template(
                'frontend.html',
                new_data=input_df.to_html(classes='dataframe', index=False),
                prediction=f"The predicted winning party is: {predicted_party}",
                probabilities=prob_df.to_html(classes='dataframe', index=False)
            )
        except Exception as e:
            return render_template('frontend.html', error=f"Error: {str(e)}")

    # On GET, just show the form
    return render_template('frontend.html')


if __name__ == '__main__':
    app.run(debug=True)
