from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model_weights.pkl", "rb"))


@app.route('/')
def use_template():
    return render_template("index.html")
# 'age', 'sex', 'bmi', 'children', 'smoker', 'age_range','have_children'
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_one = request.form['1']
    input_two = request.form['2']
    input_three = request.form['3']
    input_four = request.form['4']
    input_five = request.form['5']
    input_six = request.form['6']
    input_seven = request.form['7']

    setup_df = pd.DataFrame([input_one, input_two, input_three, input_four, input_five, input_six, input_seven])
    charges_prediction = model.predict(setup_df)
    output_prediction = '{0:.{1}}'.format(charges_prediction[0][1], 2)
    output_prediction = str(float(output_prediction)*100)+'%'
    return render_template('result.html', pred=f'The predicted charges are: {output_prediction}')

if __name__ == '__main__':
    app.run(debug=True)
