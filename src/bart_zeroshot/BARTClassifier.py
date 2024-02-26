import os
import yaml
import json
import logging
from tqdm import tqdm
from transformers import pipeline

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read the YAML file
with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get the values from the config dictionary
MODEL_NAME = config['model_name']
DATA = config["data_path"]
MODEL_PATH = config["model_path"]
OUTPUT_PATH = config["output_path"]


def get_model_file_name(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith(".safetensors"):
                return os.path.join(path, file)
    else:
        return None
    

def load_data(path):
    logging.info(f"Loading data from {path}")
    
    # Check file type. If not json, throw an error
    for file in os.listdir(path):
        if file.endswith(".json"):
            # Load the json file
            with open(os.path.join(path,file), 'r') as file:
                data = json.load(file)

    # If there is no data loaded, throw an error
    if not data:
        raise ValueError(f"No data found in {path}")
    else:
        return data


def classify():
    logging.info("Starting the main function")

    # Save the model to MODEL_PATH. If the model already exists, skip the download and load model
    if get_model_file_name(MODEL_PATH):
        logging.info(f"Loading model from {MODEL_PATH}")
        classifier = pipeline("zero-shot-classification", MODEL_PATH)
    else:
        logging.info(f"Downloading and saving model to {MODEL_PATH}")
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
        classifier.save_pretrained(MODEL_PATH)

    # Load json data
    data = load_data(DATA)
    prediction_data = data  #optional: may make a deep copy of the data

    #TODO: candidates can be given in runtime, or read from a config file
    sentiment_candidates = ["positive","negative","neutral"]
    intent_candidates = ["question","adress","request","greeting","goodbye","other"]

    # Classify the data
    for idx, element in enumerate(tqdm(prediction_data["conversation"])):
        logging.info(f"Classifying element {idx+1}/{len(prediction_data['conversation'])}")
        
        sentiment_classification = classifier(element["message"], sentiment_candidates)
        sentiment_label = sentiment_classification['labels'][0]
        prediction_data["conversation"][idx]["sentiment"] = sentiment_label

        intent_classification = classifier(element["message"], intent_candidates)
        intent_label = intent_classification['labels'][0]
        prediction_data["conversation"][idx]["intent"] = intent_label

    # Write the output to a file
    output_file = os.path.join(OUTPUT_PATH, "output.json")
    with open(output_file, 'w') as file:
        json.dump(prediction_data, file, indent=4)
    
    logging.info(f"Output file written to: {output_file}")


if __name__ == "__main__":
    classify()
    print("Process completed successfully.")
