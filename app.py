from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model_data = joblib.load('random_forest_model.pkl')
rf_model = model_data['model']
feature_order = model_data['feature_order']

# Prepares the data for the model
def prepare_input_data(form_data):
    data = {
        'population_est': [form_data['population']],
        'edu_attainment_rate': [form_data['education']],
        'unemp_rate': [form_data['unemployment']]
    }

    input_df = pd.DataFrame(data)
    input_df = input_df[feature_order]
    return input_df

# Flask code
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Gets values from forms
            population = float(request.form.get('population', 0))
            education = float(request.form.get('education', 0))
            unemployment = float(request.form.get('unemployment', 0))

            input_data = {
                'population': population,
                'education': education,
                'unemployment': unemployment
            }

            input_df = prepare_input_data(input_data)

            prediction = rf_model.predict(input_df)
            predicted_party = "Democrat" if prediction[0] == 0 else "Republican"

            prediction_prob = rf_model.predict_proba(input_df)
            prob_df = pd.DataFrame(prediction_prob, columns=['DEMOCRAT', 'REPUBLICAN'])

            return render_template(
                'frontend.html',
                new_data=input_df.to_html(classes='dataframe', index=False),
                prediction=f"The predicted winning party is: {predicted_party}",
                probabilities=prob_df.to_html(classes='dataframe', index=False)
            )
        except Exception as e:
            return render_template('frontend.html', error=f"Error: {str(e)}")

    return render_template('frontend.html')

# Main method
if __name__ == '__main__':
    app.run(debug=True)
