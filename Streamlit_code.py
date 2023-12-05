import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import sklearn
import sklearn.manifold
import cv2
import plotly.express as px
st.set_page_config(layout="wide")
script_path = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def loading_model():
    return tf.keras.saving.load_model(script_path+'/model/model_vgg16_shuffle_edit_weights_2.hdf5')
model=loading_model()
@st.cache_data
def load_props_data():
    dat=pd.read_csv('D:/CSLWB/data for app/all_props_grouped_and_with_transform.csv')
    propcols=[t for t in dat.columns if 'prop_' in t]
    transformer = sklearn.manifold.Isomap(n_components=2)
    X_transformed = transformer.fit_transform(dat[propcols][dat['is_region']==1])
    grouped=transformer.transform(dat[propcols][dat['is_region']==0])
    dat.loc[dat['is_region']==1,'C1']=X_transformed[:,0]
    dat.loc[dat['is_region']==1,'C2']=X_transformed[:,1]
    dat.loc[dat['is_region']==0,'C1']=grouped[:,0]
    dat.loc[dat['is_region']==0,'C2']=grouped[:,1]
    dat['annotation']=dat['annotation'].fillna(' ')
    dat['symbol']=0
    return dat,transformer
    
def create_data_for_scatter(dat,transformer,newvals):
    new_data=pd.DataFrame(columns=dat.columns)
    propcols=[t for t in dat.columns if 'prop_' in t]
    new_data.loc[0,propcols]=newvals
    new_data[propcols]=new_data[propcols].astype(float)
    new_data['urban']=new_data['prop_3']+new_data['prop_4']+new_data['prop_8']
    new_data['developed']=new_data['prop_3']+new_data['prop_4']+new_data['prop_8']+new_data['prop_7']
    new_data['nature']=new_data['prop_1']+new_data['prop_2']+new_data['prop_5']+new_data['prop_6']
    new_data['is_region']=0
    new_data['annotation']=' '
    new_data['size']=20
    new_data['city']=' '
    new_data['name']=' '
    new_data['color']=10
    new_tr=transformer.transform(new_data[propcols])
    new_data.loc[:,'C1']=new_tr[:,0]
    new_data.loc[:,'C2']=new_tr[:,1]
    tempdat=pd.concat([dat,new_data],ignore_index=True)
    tempdat.loc[len(tempdat)-1,'symbol']=17
    tempdat.loc[len(tempdat)-1,'annotation']='YOU ARE HERE!!!'
    return tempdat
    
def color(image):
    color_map = {0: np.array([0, 0, 0]),
             1: np.array([128, 0, 0]),
             2: np.array([0, 255, 36]),
             3: np.array([148, 148, 148]),
             4: np.array([255, 255, 255]),
             5: np.array([34, 97, 38]),
             6: np.array([0, 69, 255]),
             7: np.array([75, 181, 73]),
             8: np.array([222, 31, 7])}

    data_3d = np.ndarray(shape=(image.shape[0], image.shape[1], 3), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            data_3d[i][j] = color_map[image[i][j]]
    return data_3d

dat,transformer=load_props_data()
st.title('City Sustainability')

tab1, tab2, tab3 = st.tabs(["Home page", "Upload an image", "Choose a location"])

with tab1:
    with st.sidebar:
        st.write('sidebar')

with tab2:
    
    uploaded_file = st.file_uploader('Upload a satellite image (png format)')
    if uploaded_file is not None:
        st.image(uploaded_file)
        image_path=script_path+'/test_data/'+uploaded_file.name
        if st.button('Predict the mask'):
            #image=tf.io.read_file(image_path)
            
            
            #image=tf.io.decode_image(image, channels=3, expand_animations = False)
            image=tf.io.decode_image(cv2.imdecode(np.frombuffer(uploaded_file.read()), 1), channels=3, expand_animations = False)
            image=tf.image.resize(image, (256,256), method='bilinear')
            image=tf.cast(image,dtype=tf.dtypes.uint8)
            image_norm=image/255


            model=loading_model()
            st.balloons()
            y_pred=model.predict(image_norm.numpy().reshape(1,256,256,3))
            pred=color(np.argmax(y_pred.reshape(256,256,9),axis=2))

            colors = ['#000000','#800000','#00FF24','#949494','#FFFFFF','#226126','#0045FF','#4BB549','#DE1F07']
            labels=['Background','Bareland','Rangeland','Developed space','Road','Tree','Water','Agriculture land','Building']

            unique, counts = np.unique(np.argmax(y_pred.reshape(256,256,9),axis=2), return_counts=True)
            count={k: v for k, v in zip(unique, (counts/sum(counts))*100)}
            for i in range(9):
                count.setdefault(i,0)
            count.update((x, (y/sum(counts))*100) for x, y in count.items())
            count=dict(sorted(count.items()))

            fig1, ax1 = plt.subplots()
            patches, text= ax1.pie(count.values(), labels=count.keys(), startangle=90, counterclock=False, colors=colors)
            ax1.legend(patches, labels, loc='best',bbox_to_anchor=(1, 1),fontsize=10)

            c1, c2= st.columns(2)
            c1.image(np.array(image),width=350)
            c2.image(pred,width=350)
            st.pyplot(fig1)
            

with tab3:
    choice=st.radio("Search for a satellite image by:",options=["city name", "latitude/longitude"])
    if choice == "city name":
        center=st.text_input('Insert here the city you would like to analyze:')
    else:
        center=st.text_input('Insert here the latitude and longitude of the area of interest separated by a comma or a space:')

    url = "https://maps.googleapis.com/maps/api/staticmap?center="+center+"&zoom=17&size=512x512&maptype=satellite&key=AIzaSyACWoZYiDF8HRT6NST2pW9_yWf1ouIIBIw"
    r = requests.get(url)
    i = Image.open(BytesIO(r.content)).convert("RGB")
    st.image(i)
    if st.button('Predict your city`s mask'):
            #image=tf.io.read_file(image_path)
            #image=tf.io.decode_image(image, channels=3, expand_animations = False)
            image=tf.image.resize(np.array(i), (256,256), method='bilinear')
            image=tf.cast(image,dtype=tf.dtypes.uint8)
            image_norm=image/255

            model=loading_model()
            st.balloons()
            y_pred=model.predict(image_norm.numpy().reshape(1,256,256,3))
            pred=color(np.argmax(y_pred.reshape(256,256,9),axis=2))

            colors = ['#000000','#800000','#00FF24','#949494','#FFFFFF','#226126','#0045FF','#4BB549','#DE1F07']
            labels=['Background','Bareland','Rangeland','Developed space','Road','Tree','Water','Agriculture land','Building']

            unique, counts = np.unique(np.argmax(y_pred.reshape(256,256,9),axis=2), return_counts=True)
            arcount=np.zeros(8)
            for ind in range(len(unique)):
                if unique[ind]!=0:
                    arcount[unique[ind]-1]=counts[ind]
            arcount=arcount/np.sum(arcount)
            
            count={k: v for k, v in zip(unique, (counts/sum(counts))*100)}
            for i in range(9):
                count.setdefault(i,0)
            count.update((x, (y/sum(counts))*100) for x, y in count.items())
            count=dict(sorted(count.items()))
            
            tempdat=create_data_for_scatter(dat,transformer,arcount)
            fig=px.scatter(tempdat,'C1','C2',color='nature',size='size',symbol='symbol',text='annotation',width=1000,height=900,color_continuous_scale='rainbow')
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=False)
            p_x=tempdat.loc[len(tempdat)-1,'C1']
            p_y=tempdat.loc[len(tempdat)-1,'C2']
            fig.add_annotation(x=-0.7, y=0.63,text="YOU ARE HERE!",showarrow=False)   
            fig.add_annotation(ax=-0.7, ay=0.6,axref="x", ayref="y",x=p_x,y=p_y,showarrow=True,arrowsize=2,arrowhead=1,xanchor="right",yanchor="top")
            st.plotly_chart(fig, use_container_width=True)
            #fig1, ax1 = plt.subplots()
            #patches, text= ax1.pie(count.values(), labels=count.keys(), startangle=90, counterclock=False, colors=colors)
            #ax1.legend(patches, labels, loc='best',bbox_to_anchor=(1, 1),fontsize=10)

            c1, c2= st.columns(2)
            c1.image(np.array(image),width=350)
            c2.image(pred,width=350)
            st.pyplot(fig1)
            
