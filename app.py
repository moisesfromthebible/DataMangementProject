from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        population = int(request.form.get('population', 0))
        education = int(request.form.get('education', 0))
        unemployment = int(request.form.get('unemployment', 0))

        party_A_percentage_2020 = (education / 5000) * 50 + (100 - unemployment) * 0.5
        party_B_percentage_2020 = 100 - party_A_percentage_2020

        prediction = (
            f"Predicted outcome based on your inputs: "
            f"Party A - {party_A_percentage_2020:.2f}% vs Party B - {party_B_percentage_2020:.2f}%"
        )

        return render_template('frontend.html', prediction=prediction)

    return render_template('frontend.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
