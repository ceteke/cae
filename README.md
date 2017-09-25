# Tensorflow Convolutional Auto-Encoder
### Adjusting Layers  
Layer format is like the following: ```(16)5c-(32)3c-2p```, in which ```(16)5c``` denotes convolution layer with 16 feature maps
while kernel size being set to 5. ```2p``` denotes 2 × 2 pooling layer.  
### Auto-Encoder
Encoder is formed using the given layer structure. Decoder is formed using the inverse of given structure.
Note that pooling layers are converted to ```max_unpooling``` using the ```argmax``` of the pooling 
layer in the encoder. If pooling layer is not available in an encoding layer, decoder does not unpool. After unpooling -if exists- 
a new convolution layer is applied in the decoder -not deconvolution-.  

L2 losses between corresponding layers of encoder and decoder is called middle error. Recosntruction and these
errors are added to get the total loss. Gradient descent is used for backpropogation.  

For details see the reference.  

**Reference:** https://arxiv.org/pdf/1506.02351.pdf