import torch
import torch.nn.functional as F
import argparse
from model import Model
from data import Data
import hydra
from hydra import compose, initialize

class Predictor:
    def __init__(self , cfg , checkpoint_path):
        self.model = Model.load_from_checkpoint(checkpoint_path , cfg = cfg)
        self.model.eval()
        self.model.freeze()
        self.data    = Data(cfg)
        self.labels  = ["sadness" , "joy"  , "love" ,
                        "anger"   , "fear" , "surprise"]

    def predict(self , text):
        inference_sample = {"text" : text}
        model_input = self.data.tokenize_data(inference_sample)
        logits = self.model(torch.tensor([model_input["input_ids"]]) ,
                            torch.tensor([model_input["attention_mask"]]))
        scores = F.softmax(logits[0] , dim = 0).tolist()
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
        predictor = Predictor(cfg, checkpoint_path = "./models/best-checkpoint.ckpt")
        print(predictor.predict(sentence))
