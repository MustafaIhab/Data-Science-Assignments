from flask import Flask, render_template
from flask.globals import request
import joblib
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

app = Flask(__name__)
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')
le = joblib.load('label_encoder.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():

    user_input = list()
    #user_input.append(request.args.get('name'))
    user_input.append(305)
    user_input.append(request.args.get('views'))
    user_input.append(request.args.get('dislikes'))
    user_input.append(request.args.get('comments'))
    user_input.append(request.args.get('disComments', 0))
    user_input.append(request.args.get('disRating', 0 ))
    user_input.append(request.args.get('err', 0))
    user_input.append(request.args.get('Category'))
    user_input[7] = le.transform([user_input[7]])[0]

    likesPred = int(model.predict(scaler.transform([user_input]))[0])
    return( 'The model predicted {} likes'.format(str(likesPred)))

if __name__ == '__main__':
    app.run(debug=True)