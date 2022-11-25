from flask import Flask, request, jsonify, redirect, url_for, render_template, session
from torch_utils import get_encoding, get_prediction
from torch_utils_xlm import xlm_get_encoding, xlm_get_prediction
from flask_session import Session

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['comment']:
            session['comment'] = request.form['comment']
            # print(session.get('comment'))
            return redirect(url_for('predict'))
    else:
        return render_template('home.html')


@app.route('/predict')
def predict():
    # get sentence, tokenize, predict, return prob
    # print('=====================')
    # print(session.get('comment'))
    # print('======================')
    sentence = session.get('comment')
    input_ids, attention_mask = get_encoding(sentence)
    xlm_input_ids, xlm_attention_mask = xlm_get_encoding(sentence)
    prob = get_prediction(input_ids, attention_mask)
    xlm_prob = xlm_get_prediction(xlm_input_ids, xlm_attention_mask)
    mbert_response = prob.tolist()
    xlmr_response = xlm_prob.tolist()
    print(mbert_response)
    if mbert_response[0][0] > mbert_response[0][1]:
        mbert_result = 'The comment is non-toxic with probability ' + str(round(mbert_response[0][0], 3))
    else:
        mbert_result = 'The comment is toxic with probability ' + str(round(mbert_response[0][1], 3))
    if xlmr_response[0][0] > xlmr_response[0][1]:
        xlmr_result = 'The comment is non-toxic with probability ' + str(round(xlmr_response[0][0], 3))
    else:
        xlmr_result = 'The comment is toxic with probability ' + str(round(xlmr_response[0][1], 3))
    # print(response_obj)
    # except:
    #     response = jsonify({
    #         'error': 'something is wrong :('
    #     })
    return render_template('output.html', comment = sentence, mbert_result = mbert_result, xlmr_result = xlmr_result)


if(__name__ == '__main__'):
    app.run(debug=True)
