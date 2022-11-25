import torch
from PIL import Image
from xlmrTrainer import Preprocessor, Classifier

XLMR_CHECKPOINT = 'xlm-roberta-base'
XLMR_TOKEN_LENGTH = 200

# load model
xlmr = Classifier(XLMR_CHECKPOINT)
xlmr.load_state_dict(torch.load(
    '../../saved_models/xlm-roberta-base-87.51.pt'))

# preprocess sentence
preprocessor = Preprocessor(XLMR_CHECKPOINT, XLMR_TOKEN_LENGTH)


def xlm_get_encoding(sentence):
    encoded_sent = preprocessor.process_one(sentence)
    return encoded_sent['input_ids'].reshape(1, XLMR_TOKEN_LENGTH), encoded_sent['attention_mask'].reshape(1, XLMR_TOKEN_LENGTH)


def xlm_get_prediction(input_ids, attention_mask):
    prob = xlmr.predict_prob(input_ids, attention_mask)
    return prob
