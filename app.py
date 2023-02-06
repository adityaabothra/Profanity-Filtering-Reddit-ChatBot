import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from googlesearch import search
from datetime import datetime
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
import urllib.parse

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens', device="cpu")
#query_and_responses = []

from keras.models import load_model
my_model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

print("classes : ",classes)

def clean_up_sentence(sentence):
    sentence.replace("?", " ")
    sentence.replace("!", " ")
    sentence.replace(",", " ")
    sentence.replace(".", " ")
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ['?']]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
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
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = my_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.995
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    print("results",results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def special_processing(msg, result):
    """add some special handling for ... and Date Time"""

    if result not in ['...','Date and Time']:
        return result

    if result == 'Date and Time':
        return datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    result = 'URL:' + list(search(f"{msg}", 1,"en"))[0]

    return result

def invoke_solr_api(msg, topic, query_and_responses):
    print("topic:",topic)

    if topic == "All":
        query = "body" + ':(' + msg + ')'
    else:
        query = "body" + ':(' + msg + ')' + " AND " + "topic" + ':"' + topic + '"'

    query = urllib.parse.quote(query)

    inurl = "http://localhost:8983/solr/reddit/select?indent=true&q.op=OR&q=" + query + "&fl=id%2Cscore%2Cbody&defType=edismax&wt=json&indent=true&rows=10"

    print(inurl)

    data = urllib.request.urlopen(inurl)

    docs = json.load(data)['response']['docs']

    for doc in docs:
        print(f"Appending {doc['body']}")
        query_and_responses.append(doc['body'])

    return query_and_responses

def fetch_solr(msg, topic):
    print("topic:",topic)
    query_and_responses = []

    msg_words = clean_up_sentence(msg)

    msg = ' '.join(msg_words)

    print("msg:", msg)

    msg.replace('?', ' ')

    print("msg:", msg)

    query_and_responses.append(msg)

    query_and_responses = invoke_solr_api(msg, topic, query_and_responses)

    if len(query_and_responses) == 1:
        return "Sorry, I don't understand!"

    sentence_embeddings = model.encode(query_and_responses)
    print(sentence_embeddings.shape)

    cosine_similarity_scores = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:])

    print(cosine_similarity_scores)

    max_score = 0
    index_of_max = 0

    print(cosine_similarity_scores[0].tolist())

    for i,score in enumerate(cosine_similarity_scores[0].tolist()):
        print("i score", i,score)
        if score >= max_score:
            index_of_max = i
            max_score = score
    
    print("index_of_max", index_of_max)

    return query_and_responses[index_of_max+1]


def getResponse(msg, ints, intents_json, topic):
    if len(ints)==0:
        return fetch_solr(msg, topic)
        #return "Sorry I don't understand!"
    if ints[0]['intent'] == "reddit":
        return fetch_solr(msg, topic)

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    result = special_processing(msg, result)
    return result

def chatbot_response(msg, topic):
    ints = predict_class(msg, model)
    res = getResponse(msg, ints, intents, topic)
    return res

def stats_visualize():
    # Creating dataset
    topics = ['All', 'Politics', 'Environment',
            'Technology', 'Healthcare', 'Education']
    
    with open('static/topic_dist.txt') as f: dist_list = [int(line.strip()) for line in f]

    colors = ( "orange", "cyan", "brown",
            "grey", "lightgreen", "beige")

    # Creating autocpt arguments
    def func(pct):
        return "{:.1f}%\n".format(pct)

    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "black" }
    
    # Creating plot
    fig,ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(dist_list,
                                    autopct = lambda pct: func(pct),
                                    #explode = explode,
                                    labels = topics,
                                    shadow = True,
                                    colors = colors,
                                    startangle = 90,
                                    wedgeprops = wp,
                                    textprops = dict(color ="black"))

    
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Chatbot Response Topic Distribution", dict(fontweight="bold", color="black"))

    date_and_time = time.strftime("%Y%m%d-%H%M%S")

    plt.savefig('static/images/stats'+date_and_time+'.png', bbox_inches='tight')

    with open('static/relevance.txt') as f: relevance_list = [float(line.strip()) for line in f]

    # assign data
    data = pd.DataFrame({'Topics':['Politics', 'Environment', 'Technology', 'Healthcare', 'Education'],
                        #'Relevance':[0.67,0.5, 0.56, 0.38, 0.43]
                        'Relevance':relevance_list
                        })
    
    # compute percentage of each format
    percentage = []
    for i in range(data.shape[0]):
        pct = data.Relevance[i] * 100
        percentage.append(round(pct,2))
    data['Percentage'] = percentage
    
    # depict illustration
    plt.figure(figsize=(8,8))
    colors_list = ['Red','Orange', 'Blue', 'Purple', 'LightGreen']
    graph = plt.bar(data.Topics,data.Relevance, color = colors_list)
    plt.title('Average Relevance of Responses for Each Topic', dict(fontweight="bold", color="black"))
    
    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                y+height*1.01,
                str(data.Percentage[i])+'%',
                ha='center',
                weight='bold')
        i+=1
    # plt.show()

    plt.savefig('static/images/stats2'+date_and_time+'.png', bbox_inches='tight')

    return date_and_time


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    topic = request.args.get('topic_selected')

    return chatbot_response(userText,topic)

@app.route("/stats")
def get_bot_stats():
    return stats_visualize()

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
