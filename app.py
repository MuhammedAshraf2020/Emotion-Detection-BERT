from flask import Flask,render_template,url_for,request , redirect
import time
from hydra import compose, initialize
from onnx_inference import Predictor

app = Flask(__name__)

with initialize(config_path="./configs", job_name="BERT"):
        cfg = compose(config_name="config")
        predictor = Predictor(cfg, model_path = "./models/model.onnx")

@app.route('/' , methods=['GET' , 'POST'])
def home():
    sentence = ''
    all_time = 0
    emojis_colors = {"sadness" : "gray" ,
                     "joy"     : "gray" ,
                     "love"    : "gray" ,
                     "anger"   : "gray" ,
                     "fear"    : "gray" ,
                     "surprise": "gray" }

    if request.method == 'POST':
        start = time.time()
        sentence = request.form['sentence']
        pred =  predictor.predict(sentence)
        pred = sorted(pred , key = lambda x: x["scores"] , reverse=True)
        all_time = round((time.time() - start) * 1000)
        emojis_colors[pred[0]['label']] = 'orange'
        return render_template('index.html', sentence = sentence , all_time = all_time ,sad = emojis_colors["sadness"] , joy = emojis_colors["joy"] ,
                                              love = emojis_colors["love"]   , anger = emojis_colors["anger"] ,
                                              fear = emojis_colors["fear"]   , surprise = emojis_colors["surprise"])
    else:
        return render_template('index.html' , sentence = sentence)


if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000)
