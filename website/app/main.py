from flask import Flask, request, jsonify, render_template
from torch_utils import get_encoding, get_prediction

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # get sentence, tokenize, predict, return prob
    if request.method == 'POST':
        if request.form['sentence'] != "":
            return jsonify({
                'error': 'no input'
            })
        try:
            input_ids, attention_mask = get_encoding(request.form['sentence'])
            prob = get_prediction(input_ids, attention_mask)
            return jsonify({
                'result': prob.tolist()
            })
        except:
            return jsonify({
                'error': 'something is wrong :('
            })


if(__name__ == '__main__'):
    app.run(debug=True)
