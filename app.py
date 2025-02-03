from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load pre-trained models and encoders
rf_model = joblib.load('random_forest_titanic.pkl')
le_sex = joblib.load('label_encoder_sex.pkl')
le_embarked = joblib.load('label_encoder_embarked.pkl')

@app.route('/')
def index():
    return render_template('index.html', form_data=None, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        pclass = int(request.form['pclass'])
        sex = le_sex.transform([request.form['sex']])[0]
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = le_embarked.transform([request.form['embarked']])[0]

        # Prepare input data for the model
        input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]

        # Make prediction
        prediction = rf_model.predict(input_data)[0]

        # Prepare result message with emoji
        result = "Survived! 😊" if prediction == 1 else "Not Survived 😢"

        # Pass form data back to the template
        form_data = {
            'pclass': request.form['pclass'],
            'sex': request.form['sex'],
            'age': request.form['age'],
            'sibsp': request.form['sibsp'],
            'parch': request.form['parch'],
            'fare': request.form['fare'],
            'embarked': request.form['embarked']
        }

        return render_template('index.html', prediction=result, form_data=form_data)

    except Exception as e:
        # Handle errors gracefully and retain form data
        form_data = request.form.to_dict()
        return render_template('index.html', prediction=f"Error: {str(e)}", form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
