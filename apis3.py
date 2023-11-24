import os
import transformers
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask
from flask import request
import botocore
from boto3.session import Session

# Set environnment variables
MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_bucket = os.getenv("S3_BUCKET")
session = Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# 连接到minio
s3 = session.resource('s3', endpoint_url=MLFLOW_ROUTE)

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

# Loading model
print("Loading model from: {}".format(MLFLOW_ROUTE))


MLFLOW_MODEL = os.getenv("MLFLOW_MODEL")
download_s3_folder(s3_bucket, MLFLOW_MODEL, "./servepath")

# Creation of the Flask app
app = Flask(__name__)

# API 
# Flask route so that we can serve HTTP traffic on that route
@app.route('/',methods=['POST', 'GET'])
# Return predictions of inference using Iris Test Data
def prediction():
    question = "placeholder"
    if request.method == 'POST':
       request_data = request.get_json()
       question = request_data['question']
    else:
       question = "Which kubernetes should should I use? Red Hat or VMWare?"
    tokenizer = AutoTokenizer.from_pretrained("./servepath/components/tokenizer")
    model = AutoModelForCausalLM.from_pretrained("./servepath/model")
 
    new_user_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    mes = format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

    return {'response': mes}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
