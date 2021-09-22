from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils.fr_utils import *
from utils.inception_blocks_v2 import *
from pathlib import Path
from pymongo import MongoClient

np.set_printoptions(threshold=np.inf, linewidth=np.nan)



def triplet_loss(y_true, y_pred, alpha = 0.2):
     
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist- neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
       
    return loss

# with tf.compat.v1.Session() as test:
#     tf.random.set_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random.normal ([3, 128], mean=6, stddev=0.1, seed = 1),
#               tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
#               tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
#     loss = triplet_loss(y_true, y_pred)
    
#     print("loss = " + str(loss.eval()))    

# def compile_model(FRmodel):
#     FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
#     load_weights_from_FaceNet(FRmodel)

def facereco_run():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    print("Done")
    return FRmodel


def mongo_connection(connection_string):
    try:
        client = MongoClient(connection_string)
        db = client.get_database('face_recog_app')
        records = db.ImagesEmbedding
        return records
    except:
        print("Connection unsucessful")

def refresh_database(records):
    retrieved_database={}
    tmpdatabase_list = list(records.find())
    no_of_ids=len(tmpdatabase_list)

    for i in range(no_of_ids):    
        name = tmpdatabase_list[i]['name']
        arr = tmpdatabase_list[i]['embedding']
        retrieved_database[name] = np.array(arr)
    return retrieved_database


def insert_database(database, connection):
    for key, value in database.items():
        counter = connection.count_documents({})
        new_users={
            "id":counter+1,
            "name":key,
            "embedding":value.tolist()   
        }            
        connection.insert_one(new_users)
        print("Database has been uploaded")

def k(connection):
    retrieved_database={}
    tmpdatabase_list = list(connection.find())
    no_of_ids=len(tmpdatabase_list)

    for i in range(no_of_ids):    
        name = tmpdatabase_list[i]['name']
        arr = tmpdatabase_list[i]['embedding']
        retrieved_database[name] = np.array(arr)
    return retrieved_database
   
def insert_new_user(img_path, label, connection, model):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = img_path
    #img = cv2.imread(img_path)
    faces, num_detection = face_cascade.detectMultiScale2(img)
    
    if len(num_detection) > 0:
        for x, y, w, h in faces:
            face_roi = img[y:y+h, x:x+h]
    else:
        print("Face not detected")
        
    user_encoding = img_to_encoding(face_roi, model)
    counter = connection.count_documents({})
    new_user ={
        "id":counter+1,
        "name":label,
        "embedding":user_encoding.tolist()        
        }
    connection.insert_one(new_user)

def identify_user(img_path, model):
    retrieved_data = refresh_database()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = img_path
    #img = cv2.imread(img_path)
    faces, num_detection = face_cascade.detectMultiScale2(img)
    
    if len(num_detection) > 0:
        for x, y, w, h in faces:
            face_roi = img[y:y+h, x:x+h]
    else:
        print("Face not detected")
    
    encoding = img_to_encoding(face_roi, model)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in retrieved_data.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name  
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

if __name__ == "__main__":
    facereco_run()
    triplet_loss(y_true, y_pred, alpha = 0.2)
    compile_model(FRmodel)
    mongo_connection(connection_string)
    insert_database(database, connection)
    refresh_database(connection)
    



