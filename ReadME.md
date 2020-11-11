#Toxicity Detection in comments
##Requirements <br>
Keras==2.4.3 <br>
TensorFlow==2.3.0 <br>
sudo -H pip3 install streamlit <br>
sudo -H pip3 install matplotlib <br>
<br>
Training.ipynb trains the model and outputs model.h5 uses train.csv.zip and test.csv.zip. Was trainied on kaggle so there maybe some discrepancies in adding the files.<br>
 <br>
Pred.py uses model.h5 to predict the toxicity values for inputted sentence<br>
<br>
###To run<br>
streamlit run Pred.py<br>


