import pandas as pd
from src.utils import load_object

class PredictPipeline:

    def predict(self,features):

        model=load_object("artifacts/model.pkl")

        preds=model.predict(features)

        return preds