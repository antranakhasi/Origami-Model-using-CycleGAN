import torch
from torchvision.models import vgg19
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# CONFIG
IMG_SIZE = 512 # change to 256 for quick testing
LEARNING_RATE = None

CONTENT_LAYERS = None
STYLE_LAYERS = None

CONTENT_WEIGHT = None
STYLE_WEIGHT = None
NUM_STEPS = None

LAYER_INDICES = {
    'conv1_1': '0',
    'conv1_2': '2',
    'conv2_1': '5',
    'conv2_2': '7',
    'conv3_1': '10',
    'conv3_2': '12',
    'conv3_3': '14',
    'conv3_4': '16',
    'conv4_1': '19',
    'conv4_2': '21',
    'conv4_3': '23',
    'conv4_4': '25',
    'conv5_1': '28',
    'conv5_2': '30',
    'conv5_3': '32',
    'conv5_4': '34'
}

LAYER_CONFIGS = {
    'gatys': {
        'content': ['conv4_2'],
        'style': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    }
}

# select active layer config to change layers for feature extraction
ACTIVE_LAYER_CONFIG = "gatys" 

# load pre-trained model and get weights
vgg = vgg19(pretrained=True).features

# cpu or gpu?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load image and preprocess it for VGG19
def img_to_tensor(path, max_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")

    # resize
    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size) 

    # resize and normalize image for vgg19
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # mean and std taken from ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_img = transform(img).unsqueeze(0)
    return processed_img.to(device)


# convert image back to viewable from optimized tensor
def tensor_to_img(tensor):
    # get img, get rid of gradients
    img = tensor.cpu().clone().detach()
    img = img.squeeze(0)
    # de-normalize
    img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225])(img)
    img = img.clamp(0,1)
    img = transforms.ToPILImage()(img)
    return img

# feature extration 

def extract_features(tensor_img, layers, model=vgg):

    x = tensor_img
    
    # dict for layer name -> index
    layers_to_extract = {LAYER_INDICES[layer]: layer for layer in layers}

    for index, layer in model._modules.items():
        x = layer(x)
        
        if index in layers_to_extract:
            features[layers_to_extract[index]] = x
    
        # initializing features dictionary
        features={}

    return features
    
def gram_matrix(tensor_img):
    batch, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram

# gram matrix to capture style

# style transfer
# def nst():

# running the whole thing: 
if __name__ == "__main__":
    img_url = "./test_imgs/cat.jpg"

    org_img = Image.open(img_url)

    processed_img = img_to_tensor(img_url)
    converted_img = tensor_to_img(processed_img)
    print(converted_img.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(org_img)
    axes[1].imshow(converted_img)

    plt.tight_layout()
    plt.show()