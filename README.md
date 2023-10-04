# Restaurant-Chatbot
PROCESS

1.Creation of json file :
In our project with a JSON file we created a bunch of messages that the user is likely to type in and map them to a group of appropriate responses.
 The tag on each dictionary in the file indicates the group that each message belongs to.
2.String tokenizing :
 For each pattern we will turn it into a list of words using nltk.word_tokenizer, rather than having them as strings. We will then add each pattern into our docs_x list and its associated tag into the docs_y list.
 
            for intent in data['intents']: 
                  for pattern in intent['patterns']:
                      wrds = nltk.word_tokenize(pattern)
                       words.extend(wrds)
                       docs_x.append(wrds)
                       docs_y.append(intent["tag"])        
                 if intent['tag'] not in labels:
                     labels.append(intent['tag'])

3.Stemming :
We will use this process of stemming words to reduce the vocabulary of our model and attempt to find the more general meaning behind sentences.Stemming a word is attempting to find the root of the word.

           
          words = [stemmer.stem(w.lower()) for w in words if w != "?"]
           words = sorted(list(set(words)))
           labels = sorted(labels)

4.Bag of words :
We used Bag of Words  for representing each sentence with a list the length of the amount of words in our model's vocabulary since NN and ML algorithms require numerical inputs.Each position in the list will represent a word from our vocabulary.

5.Developing a model :
    Now that we have preprocessed all of our data we are ready to start creating and training a model. For our purposes we used a feed-forward neural network with two hidden layers. The goal of our network is to look at a bag of words and give a class that they belong to (one of our tags from the JSON file).


tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

6.Training and Saving the model :
     We fit our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training.Once we are done training the model we can save it to the file model.tflearn.


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

7.Making predictions :
Get some input from the user
Convert it to a bag of words
Get a prediction from the model
Find the most probable class
Pick a response from that class

