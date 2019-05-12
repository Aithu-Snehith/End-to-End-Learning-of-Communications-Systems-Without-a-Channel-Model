# End-to-End-Learning-of-Communications-Systems-Without-a-Channel-Model

This repo is the implementation of research paper: End-to-End Learning of Communications Systems Without a Channel Model

Link to the Paper: https://arxiv.org/pdf/1804.02276.pdf
Link to our Presentation: https://docs.google.com/presentation/d/1rth0ffRiQ-DXspMDzX8rtJWmlHAzE1FJl5mgVgxyzqw/edit?usp=sharing
(Presentation as pdf is attached in the repo)

### List of libraries used

- numpy
- Tenorflow
- Matplotlib
- sklearn

### Oveview

This is a End-to-End model to learn the modulation of the messages so that we can achive no-loss singal at the receiver side.

The algorithm iterates between supervised training of the receiver and reinforcement learning (RL)-based training of the transmitter.

- 2_m_model : Implementation of the algorithm with binary message string ( M = {0,1} )

- 4_m_model : implementation with message space containing 4 distinct messages equispaced between 0 and 1 <br>
( M = {0, 0.25, 0.5, 0.75 } )

- 8_m_model : implementation with message space containing 8 distinct messages equispaced between 0 and 1 <br>
( M = {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 } )

- comparition : This code is our comparition of our approach with traditional autoencoder model ( suervised model).
