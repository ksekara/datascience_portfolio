from dns import rcode
import streamlit as st  
from run import insert_new_user, mongo_connection, identify_user,triplet_loss, refresh_database
from utils.inception_blocks_v2 import faceRecoModel
from utils.fr_utils import load_weights_from_FaceNet,img_to_encoding
from PIL import Image
import numpy as npde
import cv2

st.cache()
def model_rec():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    print("Done")
    return FRmodel

st.cache()
def image_resize(image, width = None, height = 300, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]   
    if width is None and height is None:
        return image   
    if width is None:      
        r = height / float(h)
        dim = (int(w * r), height)
    else:        
        r = width / float(w)
        dim = (width, int(h * r))    
    resized = cv2.resize(image, dim, interpolation = inter)  
    return resized

con_str ="mongodb+srv://Kalindu:LM5yJx95m9EqsQDg@freecluster.gqsr9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
st.cache()
def mongo_con():
    rec = mongo_connection(con_str)
    return rec

st.cache()
def get_data():
    records = mongo_con()
    ret_data = refresh_database(records)
    return ret_data

DEMO_IMAGE ='./images\demo.png'

st.title("Welcome to Oneshot Learning Face Recognition")

st.sidebar.title("Select the option")
app_mode = st.sidebar.selectbox( "Select",
                                ['Insert User', 'Recognize User']
                                )

if app_mode == "Insert User":
    new_image = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    print(new_image)

    if new_image is not None:
        image = np.array(Image.open(new_image))
        org_image = image
        resize_image = image_resize(image)
        
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
        resize_image = image_resize(image)
    st.image(resize_image)

   
    name = st.text_input("Enter the User Name")
    sumbit_btn = st.button("Submit")
    if sumbit_btn:
        FRmodel = model_rec()
        records = mongo_con()
        insert_new_user(org_image, name, records, FRmodel)
        st.write("Upload Success")

elif app_mode == "Recognize User":
    new_image = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    print(new_image)

    if new_image is not None:
        image = np.array(Image.open(new_image))
        org_image = image
        resize_image = image_resize(image)
        
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
        resize_image = image_resize(image)
    st.image(resize_image)

    sumbit_btn = st.button("Submit")
    if sumbit_btn:
        FRmodel = model_rec()
        records = mongo_con()
        retrieved_data =refresh_database(records)
       
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = org_image
        
        faces, num_detection = face_cascade.detectMultiScale2(img)
        
        if len(num_detection) > 0:
            for x, y, w, h in faces:
                face_roi = img[y:y+h, x:x+h]
        else:
            st.write("Face not detected")
        
        encoding = img_to_encoding(face_roi, FRmodel)
        min_dist = 100     
        dis_list=[]
        name_list=[]  
        
        for (name, db_enc) in retrieved_data.items():            
            
            dist = np.linalg.norm(encoding-db_enc)            
            name_list.append(name)
            dis_list.append(dist)

        leasval_idx = np.argmin(dis_list)
        name = name_list[leasval_idx]  

        st.write(name)    
        
        



    


    

        
            

    
    
    