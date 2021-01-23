import re

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class SentimentModel:

    def __init__(self, model):
        self.model = model
        self.tokenizer = Tokenizer(num_words=8306, lower=False,
                                   char_level=False)

    def preprocess_text(self, sen: str):
        sentence = re.sub('[^a-zA-Z]', ' ', sen)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub("@[A-Za-z0-9]+", " ", sentence)
        self.tokenizer.fit_on_texts(sentence)
        training_sequence = self.tokenizer.texts_to_sequences(sentence)
        data = pad_sequences(training_sequence, maxlen=50)
        return data

    def predict(self, sentence):
        data = self.preprocess_text(sentence)
        prediction = self.model.predict(data).tolist()
        response = {
            "prediction": {
                "Worry": prediction[0][0],
                "Anger": prediction[0][1],
                "Disgust": prediction[0][2],
                "Fear": prediction[0][3],
                "Anxiety": prediction[0][4],
                "Sadness": prediction[0][5],
                "Happiness": prediction[0][6],
                "Relaxation": prediction[0][7],
                "Desire": prediction[0][8],

            }
        }

        return response
