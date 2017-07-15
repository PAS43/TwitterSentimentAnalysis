from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import pickle
import re

"""
amount of samples out to the 1 million to use, my 960m 2GB can only handel
about 30,000ish at the moment depending on the amount of neurons in the
deep layer and the amount fo layers.
"""
maxSamples = 20000

#Load the CSV and get the correct columns
data = pd.read_csv("C:\\Users\\Def\\Desktop\\Sentiment Analysis Dataset1.csv")
dataX = pd.DataFrame()
dataY = pd.DataFrame()
dataY[['Sentiment']] = data[['Sentiment']]
dataX[['SentimentText']] = data[['SentimentText']]

dataY = dataY.iloc[0:maxSamples]
dataX = dataX.iloc[0:maxSamples]


"""
here I filter the data and clean it up but remove @ tags and hyper links and
also any characters that are not alpha numeric, I then add it to the vec list
"""
vec = []
for x in dataX.iterrows():
    #Removes Hyperlinks
    zero = re.sub("(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", x[1].values[0])
    #Removes @ tags
    one = re.sub("@\\w+", '', zero)
    #keeps only alpha-numeric chars
    two = re.sub("\W+", ' ', one)
    vec.append(two)

"""
This loop looks for any Tweets with characthers shorter than 2 and once foudn write the
index of that Tweet to an array so I can remove from the dataframe of sentiment and the
list of Tweets later
"""
indexOfBlankStrings = []
for index, string in enumerate(vec):
    if len(string) < 2:
        indexOfBlankStrings.append(index)

for index in indexOfBlankStrings:
    del vec[index]

for row in indexOfBlankStrings:
    dataY.drop(row, axis=0, inplace=True)


"""
This makes a BOW model out of all the tweets then creates a
vector for each of the tweets containing all the words from
the BOW model, each vector is the same size becuase the
network expects it
"""
#Make BOW model and vectorise it
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(vec)
dim = tokenizer.texts_to_matrix(vec)

"""
Here im experimenting with multiple layers of the total
amount of words in the syllabus divided by ^2 - This
has given me quite accurate results compared to random guess's
of amount of neron's and amounts of layers.
"""
l1 = int(len(dim[0]) / 4) #To big for my GPU
l2 = int(len(dim[0]) / 8) #To big for my GPU
l3 = int(len(dim[0]) / 16)
l4 = int(len(dim[0]) / 32)
l5 = int(len(dim[0]) / 64)
l6 = int(len(dim[0]) / 128)


#Make the model
model = Sequential()
model.add(Dense(l4, input_dim=dim.shape[1]))
model.add(Dense(l5, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(l6, activation='relu'))
model.add(Dense(1, activation='relu'))

#Compile the model
model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['acc'])

##This runs the model
history = model.fit(x=dim, y=np.asarray(dataY), epochs=50, validation_split=0.1, shuffle=False)

print("Pickiling BOW model for later predictions")
##Pickle the Tokenizer so we can load it's word mappings for predictions later
pickle.dump(tokenizer, open("tokenizerFeatures.p", "wb"))
print("--BOW Model Saved")
print("Saving Model")
##This saves all the weights + biases. Which is basically our model
model.save("sentiment50EpochsDense27000Samples.h5")
print("--Model Saved")