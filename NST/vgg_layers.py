"""
Visualize each conv layer of VGG network to compare/contrast: exploration tool for deciding configurations

"""

from nst import *
import os
from PIL import Image

def explore_layers(content_path, style_path, output_dir='./layers'):
    os.makedirs(output_dir, exist_ok=True)

    content_tensor = img_to_tensor(content_path)
    style_tensor = img_to_tensor(style_path)

    save_all_layers(content_tensor, "content", output_dir)
    save_all_layers(style_tensor, "style", output_dir)
    return None

if __name__ == "__main__":
    content_path = 
    style_path = 

    explore_layers(content_path, style_path)




