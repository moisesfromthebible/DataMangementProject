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

@app.route('/about-model', methods=['GET'])
def about_model():
    # Information about the model
    model_info = {
        "Algorithm": "Random Forest Classifier",
        "Features": feature_order,
        "Hyperparameters": {
            "n_estimators": rf_model.get_params()['n_estimators'],
            "max_depth": rf_model.get_params()['max_depth'],
            "min_samples_split": rf_model.get_params()['min_samples_split'],
            "min_samples_leaf": rf_model.get_params()['min_samples_leaf']
        }
    }
    return render_template('about_model.html', model_info=model_info)

@app.route('/query-database', methods=['GET', 'POST'])
def query_database():
    if request.method == 'POST':
        # Get user input from the form
        table_name = request.form.get('table_name')
        limit = int(request.form.get('limit', 10))

        # Query the database
        conn = sqlite3.connect('election_results.db')
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Convert the query result to an HTML table
            table_html = df.to_html(classes='dataframe', index=False)
        except Exception as e:
            conn.close()
            return render_template(
                'query_database.html',
                error=f"Error querying the database: {str(e)}",
                result=None
            )

        return render_template('query_database.html', error=None, result=table_html)

    # Render the query form
    return render_template('query_database.html', error=None, result=None)


if __name__ == '__main__':
    app.run(debug=True)
