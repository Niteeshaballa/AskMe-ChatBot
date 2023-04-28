import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
from keras.models import load_model
from flask import Flask, render_template, request

#initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

#load the intents, words, classes and models
model = load_model('chatbotmodel.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    """
    This function tokenizes the sentence and uses lemmatizer to break each word into its
    root form

    :parameters: a sentence of datatype string
    :returns: the lemmatized list of words from the given sentence
    """
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    """
    This function creates a bag of words for all the words in the corpus
    and gives a list of 0's and 1's which consists of 1 if the word is in the sentence else 0

    :parameters: a sentence of datatype string
    :returns: a list of 0's and 1's called bag of words
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    """
    This function converts the given sentence into bag of words and passes it to the model to get
    the intent predictions.

    :parameters: a sentence of datatype string
    :returns: a dictionary with the predicted intent and its probability
    """
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    """

    :parameters: ints which are the predicted intents
    :parameters: intents_json which is the json file
    :returns: the random response of respective tag
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    """

    :parameters: msg given by the user
    :returns: the response
    """
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#create the flask app
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()