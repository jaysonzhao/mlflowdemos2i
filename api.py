import os
import transformers
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask

# Set environnment variables
MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
MLFLOW_MODEL = os.getenv("MLFLOW_MODEL")
# Loading model
print("Loading model from: {}".format(MLFLOW_ROUTE))
mlflow.set_tracking_uri(MLFLOW_ROUTE)

#TODO: replace with parameters
logged_model = 'runs:/ff87bd85ce624433b3945c25691f5117/chatbot'

#loaded_model = mlflow.pyfunc.load_model(logged_model)
mlflow.artifacts.download_artifacts(artifact_uri=logged_model, dst_path='servepath')

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
    tokenizer = AutoTokenizer.from_pretrained("./servepath/chatbot/components/tokenizer")
    model = AutoModelForCausalLM.from_pretrained("./servepath/chatbot/model")
 
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
