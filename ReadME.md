Requirements
Keras==2.4.3
TensorFlow==2.3.0
sudo -H pip3 install streamlit
sudo -H pip3 install matplotlib

Training.ipynb trains the model and outputs model.h5 uses train.csv.zip and test.csv.zip. Was trainied on kaggle so there maybe some discrepancies in adding the files.
 
Pred.py uses model.h5 to predict the toxicity values for inputted sentence

To run
streamlit run Pred.py


