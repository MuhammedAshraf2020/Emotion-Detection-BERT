import numpy as np
import argparse
import onnxruntime as ort
from hydra import compose, initialize
import torch.nn.functional as F
from scipy.special import softmax
from data import Data
from model import Model

class Predictor:
    def __init__(self , cfg , model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.dataloader  = Data(cfg)
        self.labels = ["sadness" , "joy"  , "love" ,
                        "anger"   , "fear" , "surprise"]

    def predict(self , text):
        inference_sample = {"text" : text}
        model_input = self.dataloader.tokenize_data(inference_sample)

        ort_inputs = {
        "input_ids" : np.expand_dims(model_input["input_ids"] , axis = 0) ,
        "attention_mask" : np.expand_dims(model_input["attention_mask"] , axis = 0)
        }

        logits = self.ort_session.run(None , ort_inputs)
        scores = softmax(logits[0][0]).tolist()
        predictions = []
        for idx , label in enumerate(self.labels):
            predictions.append({"label" : label ,
                                "scores": scores[idx]})
        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sentence' , '--sentence' , default = 'hello world' ,
                                                        required=True ,  help='input any sentence here')
    args = parser.parse_args()
    sentence = args.sentence    
    with initialize(config_path="./configs", job_name="BERT"):
        cfg = compose(config_name="config")
        predictor = Predictor(cfg, model_path = "./models/model.onnx")
        print(predictor.predict(sentence))


