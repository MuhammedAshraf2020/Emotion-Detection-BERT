FROM huggingface/transformers-pytorch-cpu:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3" , "app.py"]