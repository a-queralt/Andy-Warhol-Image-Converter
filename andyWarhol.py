import pandas as pd
import numpy as np
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import random
import streamlit as sl
import webcolors
from PIL import Image as im
from io import BytesIO

# Variables
if 'n_colours' not in sl.session_state:
    sl.session_state.n_colours = 4 # Amount of colours used in the final image
if 'n_colours_old' not in sl.session_state:
    sl.session_state.n_colours_old = 0 # Amount of colours used in previous model     
if 'updateModel' not in sl.session_state:
    updateModel = False
if 'original_img' not in sl.session_state:    
    sl.session_state.original_img = None
if 'new_image_array' not in sl.session_state:   
    sl.session_state.new_image_array = None
if 'aw_array' not in sl.session_state:
    sl.session_state.aw_array = None
if 'update_model' not in sl.session_state:
    sl.session_state.update_model = True
if 'modeled_image' not in sl.session_state:
    sl.session_state.modeled_image = None

colour1 = [213,0,55]
colour2 = [255,92,53]
colour3 = [230,66,122]
colour4 = [237,219,0]
colour5 = [97,166,14]
colour6 = [126,87,197]
colour7 = [0,161,155]
colour8 = [0,117,201]
colour_palette = np.array([colour1,colour2,colour3,colour4,colour5,colour6,colour7,colour8]) # Available colours for random selection
colour_scheme = np.array([colour1,colour2,colour3,colour4,colour5,colour6]) # Colours currently used in the picture

# Page configuration
sl.set_page_config(layout="wide")

# Functions
def confirmModelUpdate():
    sl.session_state.update_model = True

# Create K-means model to sort colours and apply to image
def model(n_colours,n_colours_old, im_df):
    # Separates the imagine's pixels into n_colours clusters, using a sample of the image's pixels 
    # and groups by colour an array indicating to which cluster each pixel in the image belogs to.
    km = KMeans(n_clusters=n_colours,init='random',n_init=10)
    image_array = np.array(im_df[["r","g","b"]])
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    km.fit(image_array_sample)
    aw_array = km.predict(image_array)
    global updateModel
    updateModel = False
    return image_array, aw_array

def image_to_df(original_img:np.ndarray[int]):
    # Convert image to dataframe
    dict = {"x":[],"y":[],"r":[],"g":[],"b":[]}
    for x in range(original_img.shape[0]):
        for y in range(original_img.shape[1]):
            dict["x"].append(x)
            dict["y"].append(y)
            dict["r"].append(original_img[x,y,0])
            dict["g"].append(original_img[x,y,1])
            dict["b"].append(original_img[x,y,1])
    oi_df = pd.DataFrame(dict)    
    return oi_df


def randomizeColours():
# Choses 6 random colors from colour_palette
    colour_indeces = random.sample(range(8),6)
    for x in range(6):
        colour_scheme[x] = colour_palette[colour_indeces[x]]


def colourImage(original_img: any,aw_array: np.ndarray[int],colour_scheme: np.ndarray[int],im_df: pd.DataFrame)->np.ndarray[int]:
    # Paints the image with the colours from colour_scheme according to the cluster 
    # each pixel is classified as in aw_array
    new_image_array = np.empty([original_img.shape[0],original_img.shape[1],3])
    yt = original_img.shape[1]
    for x in range(new_image_array.shape[0]):
        for y in range(new_image_array.shape[1]):         
            current_pixel = x*yt+y
            current_colour = colour_scheme[aw_array[current_pixel]]
            new_image_array[x,y,0] = current_colour[0]/255   
            new_image_array[x,y,1] = current_colour[1]/255
            new_image_array[x,y,2] = current_colour[2]/255
    return new_image_array

def reset_Colours():
    sl.session_state.n_colours_old  = 0

# Page layout
col4, col5 = sl.columns([4,3])

opened_file = col4.file_uploader(label="Open image to convert",type=["jpg","png","jpeg"], key = 'open_image')


if sl.session_state.new_image_array is not None:
    file_name = col5.text_input(label='Insert image name here:',value='My AndyWarholified image')
    aw_image = im.fromarray((sl.session_state.new_image_array*255).astype(np.uint8)) 
    buf = BytesIO()
    aw_image.save(buf, format="JPEG")
    byte_im = buf.getvalue()   
    dw_btn = col5.download_button(label = "Download Image!", key = 'download_button',data = byte_im,use_container_width=True,file_name=file_name+'.png',on_click=confirmModelUpdate())

col1, col2, col3 = sl.columns([1,3,3])


sl.session_state.n_colours = col1.slider(label="Number of colours",min_value=3,max_value=6,value=4, key = 'colour_slider', on_change=confirmModelUpdate())

if 'open_image' and opened_file is not None:
    base_width= 400
    base_height = 500
    sl.session_state.original_img = np.array(mpimg.imread(opened_file))
    if np.max(sl.session_state.original_img)<=1:
        sl.session_state.original_img = sl.session_state.original_img*255
    full_image = im.fromarray((sl.session_state.original_img).astype(np.uint8)) 
    if full_image.width>base_width or full_image.height>base_height:
        if full_image.width>full_image.height:
            wpercent = (base_width / float(full_image.size[0]))
            hsize = int((float( full_image.size[1]) * float(wpercent)))
            resized_image =  full_image.resize((base_width, hsize), im.Resampling.LANCZOS)
        else:
            wpercent = (float(full_image.size[1]) / base_height )
            vsize = int((float( full_image.size[0]) / float(wpercent)))
            resized_image =  full_image.resize((vsize, base_height), im.Resampling.LANCZOS)
    else:
        resized_image=full_image
    sl.session_state.update_model = sl.session_state.modeled_image != resized_image  
    sl.session_state.modeled_image = resized_image
    sl.session_state.original_img = np.array(resized_image)    
if 'colour_slider' and sl.session_state.original_img is not None:
    im_df = image_to_df(sl.session_state.original_img)           
    col2.image(sl.session_state.original_img, clamp = True)
    print('Current n_col: ',sl.session_state.n_colours)
    print('Previous n_col: ',sl.session_state.n_colours_old)
    if sl.session_state.update_model or sl.session_state.n_colours!=sl.session_state.n_colours_old: 
        print('updating slider model')
        image_array, sl.session_state.aw_array = model(sl.session_state.n_colours,sl.session_state.n_colours_old,im_df)
        print(sl.session_state.update_model)
        sl.session_state.n_colours_old = sl.session_state.n_colours
        sl.session_state.update_model = False
        print(sl.session_state.update_model)
        sl.session_state.new_image_array = colourImage(sl.session_state.original_img, sl.session_state.aw_array,colour_scheme,im_df)

if col1.button(label='Randomize Colours'):
    randomizeColours()
    if sl.session_state.original_img is not None:
        sl.session_state.new_image_array = colourImage(sl.session_state.original_img, sl.session_state.aw_array,colour_scheme,im_df)
      
subcol1, subcol2 = col1.columns(2)
r,g,b = webcolors.hex_to_rgb(subcol1.color_picker("Colour 1 :",key = 'colour_box_1', value = webcolors.rgb_to_hex(colour_scheme[0])))
colour_scheme[0] = [r,g,b]
r,g,b = webcolors.hex_to_rgb(subcol2.color_picker("Colour 2 :",key = 'colour_box_2', value = webcolors.rgb_to_hex(colour_scheme[1])))
colour_scheme[1] = [r,g,b]
r,g,b = webcolors.hex_to_rgb(subcol1.color_picker("Colour 3 :",key = 'colour_box_3', value = webcolors.rgb_to_hex(colour_scheme[2])))
colour_scheme[2] = [r,g,b]
r,g,b = webcolors.hex_to_rgb(subcol2.color_picker("Colour 4 :",key = 'colour_box_4', value = webcolors.rgb_to_hex(colour_scheme[3]),disabled=sl.session_state.n_colours<4))
colour_scheme[3] = [r,g,b]
r,g,b = webcolors.hex_to_rgb(subcol1.color_picker("Colour 5 :",key = 'colour_box_5', value = webcolors.rgb_to_hex(colour_scheme[4]),disabled=sl.session_state.n_colours<5))
colour_scheme[4] = [r,g,b]
r,g,b = webcolors.hex_to_rgb(subcol2.color_picker("Colour 6 :",key = 'colour_box_6', value = webcolors.rgb_to_hex(colour_scheme[5]),disabled=sl.session_state.n_colours<6))
colour_scheme[5] = [r,g,b]

if 'colour_box_1' or 'colour_box_2' or 'colour_box_3' or 'colour_box_4' or 'colour_box_5' or 'colour_box_6':
    if sl.session_state.new_image_array is not None:
            sl.session_state.new_image_array = colourImage(sl.session_state.original_img, sl.session_state.aw_array,colour_scheme,im_df)


if sl.session_state.new_image_array is not None:
    col3.image(image=sl.session_state.new_image_array)
