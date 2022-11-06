import numpy as np
import matplotlib.pyplot as plt

def show_particles(img=None, label='alite', output = 'image'):
    """Function to show alite and belite particles in masked format

    Parameters:
    img (numpy array) : Masked image in the form of numpy array
    
    label (str) : Label of particle to show, "alite", "belite", "others"
                  Default : "alite"
                  
    output (str) : Format of output, "image", "array"
                   Default : "image"
    Returns: image or masked array with choosen class depending upon the inputs of label and output

   """
    if label == "alite":
        if output == "array": 
            img = img*[img == 1][0]
            img[img == 1] = 1
            return img
        else: return plt.imshow(img*[img==1][0])
    elif label == "belite":
        if output == "array": 
            img = img*[img==51][0]
            img[img == 51] = 1
            return img
        else: return plt.imshow(img*[img==51][0])
    elif label == "others":
        if output == "array":
            img = img*[img == 255][0]
            img[img == 255] = 1
            return img
        else: return plt.imshow(img*[img==255][0])
    pass