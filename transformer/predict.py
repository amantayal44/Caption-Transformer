"""
Load image and predict image caption
"""

import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os
import argparse
from .model import Transformer,create_look_ahead_mask,create_padding_mask

#### HYPERPARAMETERS FOR MODEL ####
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = 5001
dropout_rate = 0.1
num_feauters = 2048
img_len = row_size*col_size
###################################

#### PATH FOR PRE TRAINED MODEL WEIGHTS ####
weights_path = 'pretrained_weights/image_caption_transformer_w_new.h5'
tokenizer_path = 'pretrained_weights/tokenizer.pickle'
###########################################

# loading image
# reshape and preprocess input for inception model
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

#image model 
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

print("... image model is loaded")

# mask for decoder
# combine padded and look ahead mask for decoder
def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

# transformer model
transformer = Transformer(num_layers=num_layer,
                          num_heads= num_heads,
                          d_model=d_model,
                          dff=dff,
                          row_size=row_size,
                          col_size=col_size,
                          target_vocab_size=target_vocab_size,
                          max_pos_encoding=target_vocab_size,
                          rate=dropout_rate)

print("... transformer model is created")

def build_model():
    """
    build model by passing random inputs
    """
    inp = tf.random.uniform((1,img_len,num_feauters))
    tar = tf.random.uniform((1,10),maxval=200,dtype=tf.int64)
    out,_ = transformer(inp,tar,False)
    print("... transformer model is build")

# load weights
def load_weights():
    transformer.load_weights(weights_path)
    print("... weights loaded")

def create_tokenizer():
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("... tokenizer created")
    return tokenizer

def evaluate_from_img_path(image):

  temp_input = tf.expand_dims(load_image(image)[0], 0)
  img_tensor_val = image_features_extract_model(temp_input)
  img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
  
  start_token = tokenizer.word_index['<start>']
  end_token = tokenizer.word_index['<end>']
   
  #decoder input is start token.
  decoder_input = [start_token]
  output = tf.expand_dims(decoder_input, 0)
  result = []

  for i in range(100):
      dec_mask = create_masks_decoder(output)
  
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)
      
      # select the last word from the seq_len dimension
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      # return the result if the predicted_id is equal to the end token
      if predicted_id == end_token:
          return result,tf.squeeze(output, axis=0), attention_weights
      result.append(tokenizer.index_word[int(predicted_id)])
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output = tf.concat([output, predicted_id], axis=-1)

  return result,tf.squeeze(output, axis=0), attention_weights

def evaluate_from_img_array(img):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img_tensor_val = image_features_extract_model(np.expand_dims(img,0))
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    
    #decoder input is start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)
    result = []

    for i in range(100):
        dec_mask = create_masks_decoder(output)
    
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token:
            return result,tf.squeeze(output, axis=0), attention_weights
        result.append(tokenizer.index_word[int(predicted_id)])
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return result,tf.squeeze(output, axis=0), attention_weights

if __name__ == "__main__":
    build_model()
    load_weights()
    tokenizer = create_tokenizer()
    path = 'images/bird.jpg'
    caption,_,_ = evaluate(path)
    print(' '.join(caption))
    
else:
    build_model()
    load_weights()
    tokenizer = create_tokenizer()
        