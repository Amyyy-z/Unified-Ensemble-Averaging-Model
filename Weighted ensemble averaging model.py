#import libraried
import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from keras.layers import Dense, concatenate

#before inputting pre-trained individual learners, input the image set for evaluation

#input pre-trained individual learners
model_A = tf.keras.models.load_model(r'C:\...\saved_model\A')
model_B = tf.keras.models.load_model(r'C:\...\saved_model\B')

model_A.trainable=False
model_B.trainable=False

#assign a prefix for suffix to one of the model
for layer in model_B.layers:
    layer._name = layer.name + str("_B") #this step is especially critical if model A and B share similar structures
    
x = model_A.input
y = model_B.input

a = model_A.output 
b = model_B.output

merged = concatenate([a*0.5173, b*0.4827]) #the numbers are changed each time based on individual learner's performance (i.e., accuracy, auc, fpr, etc.)
merged = Dense(1024, activation='relu')(merged)
merged = Dense(2, activation='softmax')(merged)
model_fusion = Model([x, y], merged) #concatenate the two models
model = model_fusion

#the fused model can be evaluated with an external dataset
    
 
