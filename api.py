import os
import transformers
import mlflow
from flask import Flask

# Set environnment variables
MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
MLFLOW_MODEL = os.getenv("MLFLOW_MODEL")
# Loading model
print("Loading model from: {}".format(MLFLOW_ROUTE))
mlflow.set_tracking_uri(MLFLOW_ROUTE)

#TODO: replace with parameters
logged_model = 'runs:/ff87bd85ce624433b3945c25691f5117/chatbot'

loaded_model = mlflow.pyfunc.load_model(logged_model)

# Creation of the Flask app
app = Flask(__name__)

# API 
# Flask route so that we can serve HTTP traffic on that route
@app.route('/',methods=['POST', 'GET'])
# Return predictions of inference using Iris Test Data
def prediction():

    mes = loaded_model.predict("Which kubernetes should should I use? Red Hat or VMWare?")

    return {'response': mes}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
