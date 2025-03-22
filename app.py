from flask import Flask, render_template
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = Flask(__name__)

# Define a function that returns some output
def my_function():

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    # Example sentence to apply BERT to
    sentence = "This is a test sentence."

    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Pass the inputs through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings (last hidden states)
    last_hidden_states = outputs.last_hidden_state

    # To get the embeddings for the entire sentence, you might take the mean of the token embeddings or the CLS token embedding
    sentence_embedding = last_hidden_states.mean(dim=1)  # Mean of all token embeddings

    # Print the resulting sentence embedding
    return sentence_embedding
@app.route('/')
def index():
    # Call the function and pass its output to the HTML template
    output = my_function()
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
