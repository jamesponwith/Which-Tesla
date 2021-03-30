
# coding: utf-8

# In[2]:

import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')


# In[3]:

import os

def get_image_paths():
    """
    Retrieves image paths from the a specified folder, ie., the photos/ directory in this case.
    
    Input:
    None
    
    Output:
    files - A list of image paths
    """
    folder = os.path.dirname("photos/")
    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files

X_img_paths = get_image_paths()
print(X_img_paths)


# In[4]:

import matplotlib.image as mpimg
import numpy as np

IMAGE_SIZE = 224

def tf_resize_images(X_img_file_paths):
    """
    Standardize the input set but resizing all images to (224,224) 
    
    Input:
    X_img_file_paths - A list of image paths that are obtained by using the get_images_paths() function. 
    
    Output:
    X_data - A Tensor of shape (None, 224, 224, 3)
    """
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), 
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data


# In[5]:

# Standardize the size of the images
X_imgs = tf_resize_images(X_img_paths)
print(X_imgs.shape)


# In[6]:

# Translate images 
move_image = tf.contrib.image.translate(X_imgs, translations=[1,1])
move_image_50 = tf.contrib.image.translate(X_imgs, translations=[50,50])
print(move_image.shape)


# In[7]:

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cv2

gs = gridspec.GridSpec(1, 6)
gs.update(wspace = 0.30, hspace = 2)

fig, ax = plt.subplots(figsize = (10, 10))

# Picture 1
plt.subplot(gs[0])
plt.imshow(cv2.cvtColor(X_imgs[0], cv2.COLOR_BGR2RGB))
plt.title('Base Image')

# Picture 2
plt.subplot(gs[1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.imshow(cv2.cvtColor(move_image[0].eval(), cv2.COLOR_BGR2RGB))
plt.title('[1,1]')

# Picture 3
plt.subplot(gs[2])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.imshow(cv2.cvtColor(move_image_50[0].eval(), cv2.COLOR_BGR2RGB))
plt.title('[50,50]')

# Picture 4
plt.subplot(gs[3])
plt.imshow(cv2.cvtColor(X_imgs[1], cv2.COLOR_BGR2RGB))
plt.title('Base Image')

# Picture 5
plt.subplot(gs[4])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.imshow(cv2.cvtColor(move_image[1].eval(), cv2.COLOR_BGR2RGB))
plt.title('[1,1]')

# Picture 6
plt.subplot(gs[5])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.imshow(cv2.cvtColor(move_image_50[1].eval(), cv2.COLOR_BGR2RGB))
plt.title('[50,50]')

plt.show()





