import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import pickle
import numpy
# import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()

# read intent json

try:
    # with open('dictionary.pickle') as d:
    #     data = pickle.load(d)
    raise stemmer
except:
    with open("intents.json") as file:
        data = json.load(file)

print(data)


try:
    with open("files/data.pickle", "rb") as f:
        words, labels, training, output, docs_x = pickle.load(f)
    # raise stemmer
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # create training and testing output
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("files/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output, docs_x), f)

tensorflow.compat.v1.reset_default_graph()

# define the keras model
model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(len(training[0]),)))

model.add(Dense(6, activation="relu"))
model.add(Dense(len(output[0]), activation="softmax"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

try:
    # saved_m = tensorflow.keras.models.load_model("my_model")
    # print(saved_m.evaluate(training, output, verbose=0))
    raise stemmer
except:
    model.fit(training, output, steps_per_epoch=int(training.shape[0]/8), epochs=1000, batch_size=8)
    model.save("my_model")
    print(model.evaluate(training, output, verbose=0))


#currently for all patterns, must do ONE AT A TIIIIME
def bag_of_words(s, words):
    bag = []
    # for phrase in docs_x:
    #     row = []
    #     if_match = [stemmer.stem(w) for w in phrase]
    #     s_words = nltk.word_tokenize(s)
    #     inp_words = [stemmer.stem(wr) for wr in s_words]
    #     for w in words:
    #         if w in if_match and w in inp_words:
    #             row.append(1)
    #         else:
    #             row.append(0)
    #     bag.append(row)

    bag = [0] * len(words)

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w, in enumerate(words):
            if w == se:
                bag[i] = 1

    return bag

def chat():
    print("Speak, peasant")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        pred = bag_of_words(inp, words)

        results = model.predict([pred])
        results_indx = numpy.argmax(results)
        if results[0][results_indx] < 0.7:
            print("sorry, I don't understand")
        else:
            tag = labels[results_indx]
            print(tag, ": ",results[0][results_indx])
            for intent in data["intents"]:
                if intent["tag"] == tag:
                    print(random.choice(intent["responses"]))


chat()