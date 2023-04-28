#import the required libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

#initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

#load the intents, words, classes and models
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    """
    This function tokenizes the sentence and uses lemmatizer to break each word into its
    root form

    :parameters: a sentence of datatype string
    :returns: the lemmatized list of words from the given sentence
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    This function creates a bag of words for all the words in the corpus
    and gives a list of 0's and 1's which consists of 1 if the word is in the sentence else 0

    :parameters: a sentence of datatype string
    :returns: a list of 0's and 1's called bag of words
    """
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    """
    This function converts the given sentence into bag of words and passes it to the model to get
    the intent predictions.

    :parameters: a sentence of datatype string
    :returns: a dictionary with the predicted intent and its probability
    """
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.10
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """
    This function takes the predicted intent and intents json file as input and returns a
    random response from the responses of that tag

    :parameters: intents_list which contains the predicted intents
    :parameters: intents_json which is the  intents json file
    :returns: a random response of datatype string
    """
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("|============= Welcome to AskMe Bot =============|")
print("|=============== Ask your any query about our website ================|")

#the program starts here by taking the input from the user in a loop until the user says bye or goodbye
while True:
    #takes the message from the user
    message = input("| You: ")
    #checks if the message is bye or goodbye and prints the bot response and breaks the loop
    if message == "bye" or message == "Goodbye":
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
        print("|===================== The Program End here! =====================|")
        break
    #else gets the response from the bot and waits for the user message
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
