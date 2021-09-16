from flask import Blueprint, request, jsonify
from logging import getLogger
from private.model.model import predict
import re

logger = getLogger('app.controller.scoring')

scoring_api = Blueprint("scoring", __name__, url_prefix="/scoring")

@scoring_api.route('/', methods=['POST'])
def score():
    req = request.get_json()
    clean_text = re.sub('[,.\s]', ' ', req['text'])
    predicted = predict(clean_text)
    return jsonify({'positive': predicted.positive, 'negative': predicted.negative, 'neutral': predicted.neutral, 'message': ''})

