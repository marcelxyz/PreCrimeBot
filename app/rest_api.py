from flask import Flask, jsonify

from query_handler import predict

app = Flask(__name__)


@app.route('/predict-crime/<date>/<address>', methods=['GET'])
def predict_crime(date, address):
    try:
        return jsonify(predict(date, address).__dict__)
    except ValueError as e:
        return jsonify(str(e)), 400


if __name__ == '__main__':
    app.run(debug=True)
