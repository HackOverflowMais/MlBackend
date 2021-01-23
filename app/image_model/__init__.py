import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class ImageModel:
    def __init__(self, model):
        self.model = model

    def model_predict(self, img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        result = int(preds[0][0])
        if result == 1:
            out = True
        else:
            out = False
        return {"covid": out}
