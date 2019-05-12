#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Aithu-Snehith/End-to-End-Learning-of-Communications-Systems-Without-a-Channel-Model/blob/master/4_pam_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### Libraries required for the implementation

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import *
from sklearn import preprocessing
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error


# ### Required Parameters (4-PAM)

# In[ ]:


msg_total = 4
channel = 8
epochs = 5000
sigma = 1e-4
batch_size = 1024


# ### Defining Required Functions

# In[ ]:


def perturbation(x):
    w = K.random_normal(shape = (channel,2),mean=0.0,stddev=sigma**0.5,dtype=None)
    xp = ((1-sigma)**0.5)*x + w
    return xp

def loss_tx(y_true, y_pred):
    return -y_true*y_pred

def get_policy(inp):
    xp = inp[0]
    x = inp[1]
    w = xp - x
    policy = -K.sum(w*w)
    return policy


# ## Modelling the Transmitter
# ### 1. Tx encoder architecture

# In[ ]:


tx_inp = Input((1,))
embbedings_layer = Dense(msg_total, activation = 'relu')(tx_inp)
layer_dense = Dense(2*channel, activation = 'relu')(embbedings_layer)
to_complex = Reshape((channel,2))(layer_dense)
x = Lambda(lambda x: keras.backend.l2_normalize(x))(to_complex)
xp = Lambda(perturbation)(to_complex)
policy = Lambda(get_policy)([xp,x])


# ### 2. Definng Models

# In[ ]:


model_policy = keras.models.Model(inputs=tx_inp, outputs=policy)
model_tx = keras.models.Model(inputs=tx_inp, outputs=xp)
model_x = keras.models.Model(inputs=tx_inp, outputs=x)

model_policy.compile(loss=loss_tx, optimizer=tf.keras.optimizers.SGD(lr = 1e-5))
print(model_policy.summary())


# ## Modelling the Receiver
# ### Rx architecture

# In[ ]:


rx_inp = Input((channel,2))
to_flat = Reshape((2*channel,))(rx_inp)
fc = Dense(8*2*channel, activation = 'relu')(to_flat)
softmax = Dense(msg_total, activation = 'softmax')(fc)

model_rx = keras.models.Model(inputs=rx_inp, outputs=softmax)

model_rx.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam())
print(model_rx.summary())


# ### Alternate Training

# In[ ]:


loss_tx = []
loss_rx = []
for epoch in range(epochs):
#     Transmitter Training
    raw_input = np.random.randint(0,msg_total,(batch_size))
    label = np.zeros((batch_size, msg_total))
    label[np.arange(batch_size), raw_input] = 1
    tx_input = raw_input/float(msg_total)
    xp = model_tx.predict(tx_input)
    y = xp + np.random.normal(0,0.001,(batch_size, channel,2))
    pred = model_rx.predict(y)
    loss = np.sum(np.square(label - pred), axis = 1)
    history_tx = model_policy.fit(tx_input, loss, batch_size=batch_size, epochs=1, verbose=0)    
    loss_tx.append(history_tx.history['loss'][0])
    
#     Receiver Training
    raw_input = np.random.randint(0,msg_total,(batch_size))
    label = np.zeros((batch_size, msg_total))
    label[np.arange(batch_size), raw_input] = 1
    tx_input = raw_input/float(msg_total)
    x = model_x.predict(tx_input)
    y = x + np.random.normal(0,0.001,(batch_size, channel,2))
    history_rx = model_rx.fit(y, label, batch_size=batch_size, epochs=1, verbose=0)
    loss_rx.append(history_rx.history['loss'][0])
    
    if(epoch % 100 == 0):
        print('epoch: ', epoch, 'tx_loss', history_tx.history['loss'][0], 'rx_loss', history_rx.history['loss'][0])


# ### Plotting Transmitter and Receiver Loss

# In[ ]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(loss_tx)
plt.title('Transmitter Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(1,2,2)
plt.plot(loss_rx)
plt.title('Receiver Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


# ### Prediction

# In[ ]:


#testing
batch_size = 100
raw_input = np.random.randint(0,msg_total,(batch_size))
print(raw_input)
label = np.zeros((batch_size, msg_total))
label[np.arange(batch_size), raw_input] = 1
tx_input = raw_input/float(msg_total)
xp = model_x.predict(tx_input)
y = xp + np.random.normal(0,0.001,(batch_size, channel,2))
pred = model_rx.predict(y)
pred_int = np.argmax(pred, axis = 1)
print(pred_int)

from sklearn.metrics import accuracy_score

print('accuracy:',accuracy_score(raw_input, pred_int))


# In[ ]:




