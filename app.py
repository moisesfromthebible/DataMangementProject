from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        population = request.form.get('population')
        education = request.form.get('education')
        unemployment = request.form.get('unemployment')

        prediction = f"Predicted outcome based on your inputs: Party A - {party_A_percentage_2020}% vs Party B - {party_B_percentage_2020}%"

        return render_template('frontend.html', prediction=prediction)

    return render_template('frontend.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
