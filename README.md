# Twitter Sentiment Analysis Model
---
### What is this repo?!

  - A model build for Twitter Sentiment analysis, P file and H5 file included
  - The Python 3 code to build the above model using Keras, Theano & Pandas

I was set the task to try and make a sentiment analysis model for Twitter for my dissertation even though the due by date is long past I still wanted to accomplish this task.

>For my dissertation I was set the task to predict
>stock prices for a single company on the stock 
>market using Twitter and sentiment analysis.
>I tried to see if you could predict if the stock
>market would go up or down given the sentiment
>of the Tweets.

### Dependancies

This model uses a number of open source projects to work properly:

* [Keras] - A Machine Learning Neural Network front end framework
* [Theano] - A Machine Learning Neural Network back end framework (For Keras)
* [Pandas] - A Python dataframe library
* [Pickle] - A Python serilisation lib, I use this to serialise the Tokenier with all the word mappings in
* [RE] - Regular expressions are not used in the version but left the method in for removing hyperlinks, at and hashtag symbols

### Training Info

I used a Cuda on a Nvidia 960M it took about 4 hours/ish for 10 epochs of 900,000 Tweets, Every 100,000 Tweets testing using 10,000 Tweets is done.

### Model Info

This model is 79% accurate and has 0.41 loss.
If you want to use remember to deseralise the Keras tokenier which is the *.p file, the predict.py has an example.
Here is a small example of the model output:
| Tweet | Score |
|------|-------|
| "fuck you, you are an asshole"| 0.3316856|
|"I love you"|0.81485116|
|"yes! I cant wait to go"| 0.60499549|
|"I don't want to be here"| 0.11294857
|"is the best person ever" | 0.74768722|

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [Theano]: <http://deeplearning.net/software/theano/>
   [Keras]: <https://keras.io/>
   [dataset]: <http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/>
   [Pandas]: <http://pandas.pydata.org/>
