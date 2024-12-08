from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

#rf_model = joblib.load('random_forest_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        population = int(request.form.get('population', 0))
        education = int(request.form.get('education', 0))
        unemployment = int(request.form.get('unemployment', 0))

        new_data = {
            'population': [population],
            'education_level': [education],
            'unemployment_rate': [unemployment]
        }
        new_data_df = pd.DataFrame(new_data)

        prediction = rf_model.predict(new_data_df)

        result = "Party A" if prediction[0] == 1 else "Party B"

        return render_template('frontend.html', prediction=f"Predicted outcome: {result}")

    return render_template('frontend.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
