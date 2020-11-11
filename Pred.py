import streamlit as st
print (st.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt


st.title('Toxicity detection in text')

@st.cache(allow_output_mutation=True)
def load_data():
    train = pd.read_csv('train.csv.zip', compression='zip', header=0, sep=',', quotechar='"')
    print ("reading done")
    trainsentences = train["comment_text"]
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(list(trainsentences))
    print ("Tokenizer done")
    loaded_model=load_model('model.h5')
    print ("Model loaded")
    return tokenizer,loaded_model


data_load_state = st.text('Getting Things Ready...')
tokenizer,loaded_model=load_data()
print ("Returned?")
data_load_state.text("Done!")

st.subheader('Enter text to analyze...')
Y=st.text_area('Enter text')
print (Y)
if Y:
	st.write('Computing')
	X=[]
	X.append(Y)
	token = tokenizer.texts_to_sequences(X)
	X_t = pad_sequences(token, maxlen=200)
	score = loaded_model.predict(X_t)
	print ("Toxic: "+ str(score[0][0]))
	print ("Severe_Toxic: "+str(score[0][1]))
	print ("Obscene: "+str(score[0][2]))
	print ("Threat: "+str(score[0][3]))
	print ("Insult: "+str(score[0][4]))
	print ("Identity_Hate: "+str(score[0][5]))
	st.subheader('Toxicity Analysis')
	data = {'Toxic':score[0][0],'Severe_Toxic':score[0][1],'Obscene':score[0][2],'Threat':score[0][3],'Insult':score[0][4],'Identity_Hate':score[0][5]} 
	courses = list(data.keys()) 
	values = list(data.values()) 
	plot = plt.figure(figsize = (10, 5)) 
	plt.bar(courses, values, color =['purple', 'indigo', 'blue','green', 'orange','red'], width = 0.4) 
	plt.ylim(0,1)
	plt.yticks([0,0.025, 0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	plt.xlabel("Categories") 
	plt.ylabel("Percentage") 
	plt.title("Toxicity Analysis") 
	#plt.show() 
	st.pyplot(fig=plot)


