'''
@Filename: web.py
@Author: nbgao (Gao Pengbing)
@Contact: nbgao@126.com
'''
import streamlit as st
import requests
import argparse
import os
import io
import time
import wget
import numpy as np
from PIL import Image

# REST_API_URL = 'http://127.0.0.1:5005/generate'
REST_API_URL = 'http://192.168.177.202:5005/generate'

def generate_result(task, image_buffer):
    # image = open(image_path, 'rb').read()
    files = {'image': image_buffer}
    payload = {'task': task}

    r = requests.post(REST_API_URL, files=files, data=payload).json()
    # Return generate result
    if r['success']:
        generate_image = r['generate']
        return generate_image
    else:
        print('Request failed')


# HOME page
def home():
    with open('home.md', 'r') as f:
        st.markdown(f.read())


# Painting->Photo page
def painting_photo():
    task = 'A2B'
    image_height = 256
    save_path = 'data/A2B'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mode = st.radio('Image path mode:', ['DIR', 'URL'])

    if mode == 'DIR':
        image_file = st.file_uploader('Load landscape painting:', type='jpg')
        if image_file is not None:
            image_buffer = image_file.read()
            image = Image.open(io.BytesIO(image_buffer))
            image_path = '{}/{}.jpg'.format(save_path, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            image.save(image_path)
            W, H = image.size
            W, H = W*image_height//H, image_height
            st.image(image.resize((W,H)))
                  
            generate_photo_arr = generate_result(task, image_buffer) 
            generate_photo = Image.fromarray(np.asarray(generate_photo_arr, np.uint8))
            # generate_photo = Image.open(io.BytesIO(generate_photo_buffer))
            st.text('Generate Photo:')
            st.image(generate_photo)

    elif mode == 'URL':
        image_url = st.text_input('Image URL:')
        if image_url not in [None, '']:
            image_type = image_url.split('.')[-1]
            image_path = '{}/{}.{}'.format(save_path, time.strftime("%Y%m%d-%H%M%S", time.localtime()), image_type)
            wget.download(image_url, image_path)
            print()
            image_buffer = open('image_path', 'rb').read()
            image = Image.open(io.BytesIO(image_buffer))
            W, H = image.size
            W, H = W*image_height//H, image_height
            st.image(image.resize((W,H)))

            generate_photo_arr = generate_result(task, image_buffer) 
            generate_photo = Image.fromarray(np.asarray(generate_photo_arr, np.uint8))
            # generate_photo = Image.open(io.BytesIO(generate_photo_buffer))
            st.text('Generate Photo:')
            st.image(generate_photo)


# Photo->Painting page
def photo_painting():
    task = 'B2A'
    image_height = 256
    save_path = 'data/B2A'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mode = st.radio('Image path mode:', ['DIR', 'URL'])

    if mode == 'DIR':
        image_file = st.file_uploader('Load landscape photo:', type='jpg')
        if image_file is not None:
            image_buffer = image_file.read()
            image = Image.open(io.BytesIO(image_buffer))
            image_path = '{}/{}.jpg'.format(save_path, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            image.save(image_path)
            W, H = image.size
            W, H = W*image_height//H, image_height
            st.image(image.resize((W,H)))
                  
            generate_painting_arr = generate_result(task, image_buffer) 
            generate_painting = Image.fromarray(np.asarray(generate_painting_arr, np.uint8))
            # generate_painting = Image.open(io.ByteIO(generate_painting_buffer))
            st.text('Generate Painting:')
            st.image(generate_painting)

    elif mode == 'URL':
        image_url = st.text_input('Image URL:')
        if image_url not in [None, '']:
            image_type = image_url.split('.')[-1]
            image_path = '{}/{}.{}'.format(save_path, time.strftime("%Y%m%d-%H%M%S", time.localtime()), image_type)
            wget.download(image_url, image_path)
            print()
            image = Image.open(io.BytesIO(open(image_path, 'rb').read()))
            W, H = image.size
            W, H = W*image_height//H, image_height
            st.image(image.resize((W,H)))
       
            generate_painting_arr = generate_result(task, image_buffer) 
            generate_painting = Image.fromarray(np.asarray(generate_painting_arr, np.uint8))
            # generate_painting = Image.open(io.ByteIO(generate_painting_buffer))
            st.text('Generate Painting:')
            st.image(generate_painting)
            
    return

if __name__ == '__main__':
    st.title('GAN Lanscape Painting-Photo Web')
    server_map = {'Home': home, 'Painting->Photo': painting_photo, 'Photo->Painting': photo_painting}
    page = st.selectbox('Inference Service', list(server_map.keys()))
    st.header(page)
    server_map[page]()