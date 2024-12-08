from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        median_income = request.form.get('median_income')
        population_density = request.form.get('population_density')
        percent_white = request.form.get('percent_white')
        percent_below_poverty = request.form.get('percent_below_poverty')
        age_0_18_percent = request.form.get('age_0_18_percent')
        age_19_35_percent = request.form.get('age_19_35_percent')
        age_65_plus_percent = request.form.get('age_65_plus_percent')
        unemployment_rate = request.form.get('unemployment_rate')
        labor_force_participation_rate = request.form.get('labor_force_participation_rate')
        voter_turnout_rate = request.form.get('voter_turnout_rate')
        party_A_percentage_2020 = request.form.get('party_A_percentage_2020')
        party_B_percentage_2020 = request.form.get('party_B_percentage_2020')
        election_year = request.form.get('election_year')

        prediction = f"Predicted outcome based on your inputs: Party A - {party_A_percentage_2020}% vs Party B - {party_B_percentage_2020}%"

        return render_template('frontend.html', prediction=prediction)

    return render_template('frontend.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
