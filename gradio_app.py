import gradio as gr
import numpy as np
from transformer.predict import evaluate_from_img_array

def caption(img):
    result,_,_ = evaluate_from_img_array(img)
    return ' '.join(result)

gr.Interface(caption,gr.inputs.Image(shape=(299,299)),"text").launch()