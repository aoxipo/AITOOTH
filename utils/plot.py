import matplotlib.pyplot as plt
import torch


def plot_rect(image, label):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    if(len(image.shape) == 3):
        width,height,channel = image.shape
    else:
        width,height = image.shape
    
    if(label is dict):
        tag = label['pred_logits']
        tag = torch.argmax(tag.view(-1, 9),1)
        label = label['pred_boxes']
    if label.device != 'cpu':
        label = label.cpu()
    if len(label.shape) == 3:
        label = label.squeeze()
    index =0
    for coord in label:
        if(len(coord) == 5):
            class_number, x,y,w,h = coord
        else:
            x,y,w,h = coord
            class_number = tag[index]

        centerx = width*(x - w/2)
        centery = height*(y - h/2)
        W = w * width
        H = h * height
        plt.gca().add_patch(
            plt.Rectangle(
                xy=(centerx,centery),
                width=W, 
                height=H,
                edgecolor='red',
                fill=False, linewidth=1
            )
        )
        plt.text(centerx, centery, '{}'.format(int(class_number)), ha='center', va='center')
    return 

def plot_rect_old(image, label):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    if(len(image.shape) == 3):
        width,height,channel = image.shape
    else:
        width,height = image.shape
    for coord in label:
        class_number, x,y,w,h= coord 
        centerx = width*(x - w/2)
        centery = height*(y - h/2)
        W = w * width
        H = h * height
        plt.gca().add_patch(
            plt.Rectangle(
                xy=(centerx,centery),
                width=W, 
                height=H,
                edgecolor='red',
                fill=False, linewidth=1
            )
        )
        plt.text(centerx, centery, '{}'.format(int(class_number)), ha='center', va='center')
    return 