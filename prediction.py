from keras.models import load_model
import pickle


model = load_model("C:/Users/Def/PycharmProjects/KerasUkExpenditure/900k Samples 100K validation 048 losss 079 acc/900KSamples50Epochs.h5")
t = open("C:/Users/Def/PycharmProjects/KerasUkExpenditure/900k Samples 100K validation 048 losss 079 acc/tokenizer.p", "rb")
file = pickle.load(t)
texts = []
texts.append("fuck you, you are an asshole")
texts.append("I love you")
texts.append("yes! I cant wait to go")
texts.append("I don't want to be here")
texts.append("is the best person ever")
# out1 = file.fit_on_texts(texts=arr)
out2 = file.texts_to_matrix(texts=texts)
pred = model.predict(out2)
print("Prediction: ", pred)