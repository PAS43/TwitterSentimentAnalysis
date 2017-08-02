from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle
import re

"""
amount of samples out to the 1 million to use, my 960m 2GB can only handel
about 30,000ish at the moment depending on the amount of neurons in the
deep layer and the amount of layers.
"""
maxSamples = 100000

#Load the CSV and get the correct columns
data = pd.read_csv("C:\\Users\\Def\\Desktop\\Sentiment Analysis Dataset1.csv")



"""
Method NOT USED
Here I filter the data and clean it up by removing @ tags, hyperlinks and
also any characters that are not alpha-numeric.
"""
def removeTagsAndLinks(dataframe):
    for x in dataframe.iterrows():
        #Removes Hyperlinks
        x[1].values[0] = re.sub("(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", str(x[1].values[0]))
        #Removes @ tags
        x[1].values[0] = re.sub("@\\w+", '', str(x[1].values[0]))
        #keeps only alpha-numeric chars
        x[1].values[0] = re.sub("\W+", ' ', str(x[1].values[0]))
    return dataframe

#Make the model
model = Sequential()
model.add(Dense(1, input_dim=100000))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

#Compile the model
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['acc'])

"""
This here will use multiple batches to train the model.
    startIndex:
        This is the starting index of the array for which you want to
        start training the network from.
    dataRange:
        The number of elements use to train the network in each batch so
        since dataRange = 1000 this mean it goes from
        startIndex...dataRange OR 0...1000
    amountOfEpochs:
        This is kinda self explanitory, the more Epochs the more it
        is supposed to learn AKA updates the optimisation algo numbers
"""

tokenizer = Tokenizer(lower=False, num_words=100000)
def getTestGen(data, dataStart=900000, dataEnd=990000):
    while True:
        y = data.iloc[:, 1][dataStart:dataEnd]
        xx = data.iloc[:, 3][dataStart:dataEnd]
        tokenizer.fit_on_texts(xx)
        for x in range(len(xx)):
            yield tokenizer.texts_to_matrix(xx.iloc[x:x+ 1000]), y.iloc[x:x + 1000]

def testGen(data, dataStart=0,dataEnd=10,epoch=10):
    tar = getTestGen(data=data)
    tt = 0
    for e in range(epoch):
        print("Epoch -- ", e, "\ttt -- ", tt)
        y = data.iloc[:, 1][dataStart:dataEnd]
        xx = data.iloc[:, 3][dataStart:dataEnd]
        tokenizer.fit_on_texts(xx)
        start = 0
        jump = 1000
        for x in range(start, dataEnd, jump):
            if (x > 0) and (x % 100000 == 0):
                test_batch = next(tar)
                h = model.test_on_batch(x=test_batch[0], y=test_batch[1])
                print("ValLoss: ", h[0], "Val_Acc: ",  h[1], "\n")
                print("Samples: ", x)
                yield tokenizer.texts_to_matrix(xx.iloc[x:x + jump]), y.iloc[x:x + jump]

            print("Samples: ", x)
            yield tokenizer.texts_to_matrix(xx.iloc[x:x + jump]), y.iloc[x:x + jump]



tgg = testGen(data, dataStart=0, dataEnd=900000, epoch=50)
a = 0

for features, labels in tgg:
    h = model.train_on_batch(x=features, y=labels)
    print("loss: ", h[0], " acc: ", h[1], " - ", "Itteration",a)
    a += 1

test = getTestGen(data)
for f, l in test:
    h = model.test_on_batch(x=f,y=l)
    print("Features: ",h[0], "Labels",h[1])


print("Pickiling BOW model for later predictions")
pickle.dump(tokenizer, open("tokenizer.p", "wb"))
print("--BOW Model Saved")
print("Saving Model")
model.save("900KSamples50Epochs.h5")
print("--Model Saved")

