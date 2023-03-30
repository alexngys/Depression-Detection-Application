import numpy as np
import pandas as pd
import keras 
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
test_split_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()

path = 'dataset/'
features_test = np.load(path + 'test_samples_reg.npz', allow_pickle=True)['arr_0']
labels_test = np.load(path + 'test_labels_reg.npz', allow_pickle=True)['arr_0']

X_test = np.array(features_test)
Y_test = np.array(labels_test)

def label_change(array): # 4-6 equal windows per subject or 5 second across all subjects maybe increase sampling rate
    for i in range(len(array)):
        if array[i] < 6: # mild or no symptoms
            array[i] = 0
        elif array[i] >=6 and array[i] <=14: # moderate symptoms
            array[i] = 1
        elif array[i] >=15 and array[i] <=19: # moderately severe symptoms
            array[i] = 1
        else: # severe symptoms
            array[i] = 1
    return array


Y_test = label_change(Y_test)

model = keras.models.load_model('my_model')

Y_prediction = model.predict(X_test)

Y_prediction = np.argmax (Y_prediction, axis = 1)
print(Y_prediction,Y_test)


#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(Y_test, Y_prediction , normalize='true')


cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = ['Non','Min'])

cm_display.plot()
plt.show()
