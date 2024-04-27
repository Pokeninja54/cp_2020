'''
import os
from PIL import Image
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--ROOT', default="./VIDIT/train/", help='path to VIDIT train dataset')
parser.add_argument('--TARGET', default="./pix2pixWMultipleDirections_to_SW", help='path to output dataset')
parser.add_argument('--RIGHT_DIRECTION', default="SW", help='direction to which relight')
parser.add_argument('--TEMPERATURE', default="4500", help='light temperature')

args = parser.parse_args()

ROOT = args.ROOT
TARGET = args.TARGET

#LEFT_DIRECTION = "E"
RIGHT_DIRECTION = args.RIGHT_DIRECTION

TEMPERATURE = args.TEMPERATURE

def getImages(scene, direction):
    Dir = os.path.join(scene, direction)
    imagesPaths = [os.path.join(Dir, f) for f in os.listdir(Dir)]
    imagesPaths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))
    images = [Image.open(x) for x in imagesPaths]
    return images


if __name__ == '__main__':

    mainScenes = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]

#  For temperatures
#    scenes = []
#    for scene in mainScenes:
#        scenes += [os.path.join(scene, f) for f in os.listdir(scene)]
#
    scenes = [os.path.join(scene, TEMPERATURE) for scene in mainScenes]

    vidit_dataset = load_dataset("Nahrawy/VIDIT-Depth-ControlNet")
    train_data = vidit_dataset["train"]


    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    for scene in scenes:
        for LEFT_DIRECTION in ['E',  'N',  'NE',  'NW',  'S',  'SE',  'SW',  'W']:

            scenetarget = os.path.join(TARGET, scene, LEFT_DIRECTION)
            os.makedirs(scenetarget)
    
            leftImages = getImages(scene, LEFT_DIRECTION)
            rightImages = getImages(scene, RIGHT_DIRECTION)

    
            imageSize = leftImages[0].size
    
            finalSize = (imageSize[0] * 2, imageSize[1])
    
            newIm = Image.new('RGB', finalSize)
            for i in range(len(leftImages)):
                filename = '_'.join(scene.split('/')[-2:]) + '_' + LEFT_DIRECTION + '_' + RIGHT_DIRECTION + '_' + str(i) + '.png'
                print(filename)
                newIm.paste(leftImages[i], (0, 0))
                newIm.paste(rightImages[i], (imageSize[0], 0))
                newIm.save(os.path.join(scenetarget, filename))
    '''
import os
from PIL import Image
import argparse
from datasets import load_dataset
import pickle 
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--TARGET', default="./pix2pixWMultipleDirections_to_SW", help='path to output dataset')
parser.add_argument('--RIGHT_DIRECTION', default="SW", help='direction to which relight')
parser.add_argument('--TEMPERATURE', default="4500", help='light temperature')

args = parser.parse_args()

TARGET = args.TARGET + '_LAB'
RIGHT_DIRECTION = args.RIGHT_DIRECTION
TEMPERATURE = args.TEMPERATURE

def getImages(left_images, right_images):
    # Convert image paths to PIL images
    left_images = [Image.open(image) for image in left_images]
    right_images = [Image.open(image) for image in right_images]
    return left_images, right_images

if __name__ == '__main__':

    vidit_dataset = load_dataset("Nahrawy/VIDIT-Depth-ControlNet")
    train_data = vidit_dataset["train"]

    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    #print("Going to load dictionary now!")
    '''
    scene_temp_to_desired = dict()
    j = 0
    for example in train_data:
        j += 1
        if j % 100 == 0:
            print("on iteration", j)
        scene = example["scene"]
        temp = example["temprature"]
        left_direction = example["direction"]
        left_image = example["image"]
        if left_direction == RIGHT_DIRECTION:
            scene_temp_to_desired[(scene,temp, RIGHT_DIRECTION)] = left_image
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(scene_temp_to_desired, f)
    '''
    with open('saved_dictionary.pkl', 'rb') as f:
        scene_temp_to_desired = pickle.load(f)
    print("made it past the first thing!")
    i = 0
    filename_to_orig_pic = dict()
    for example in train_data:
        scene = example["scene"]
        temp = example["temprature"]
        left_direction = example["direction"]
        left_image = example["image"]
        right_image = scene_temp_to_desired[(scene,temp,RIGHT_DIRECTION)]
        
        left_image = left_image.resize((256,256))
        right_image = right_image.resize((256,256))
        #imageSize = left_image.size
        new_size = 256
        finalSize = (512, 256)
        #print("The type of left image is", type(left_image))
        # Convert the image to LAB color space
        lab_left_image = cv2.cvtColor(np.array(left_image), cv2.COLOR_BGR2LAB)
        # Split the LAB image into its channels (L, A, and B)
        L_l_channel, _, _ = cv2.split(lab_left_image)
        L_l_channel = Image.fromarray(L_l_channel)

        lab_right_image = cv2.cvtColor(np.array(right_image), cv2.COLOR_BGR2LAB)

        # Split the LAB image into its channels (L, A, and B)
        R_l_channel, _, _ = cv2.split(lab_right_image)
        R_l_channel = Image.fromarray(R_l_channel)
        #print("The Final Size is", finalSize)
        #scenetarget = os.path.join(TARGET, scene, left_direction)
        
        filename = '_'.join(scene.split('/')[-2:]) + '_' + left_direction + '_' + RIGHT_DIRECTION + '_' + str(temp) + '.png'
        filename_to_orig_pic[filename] = left_image
        #print("Scenetarget, filename are", scenetarget, filename)
        newIm = Image.new('RGB', finalSize)
        newIm.paste(L_l_channel, (0, 0))
        newIm.paste(R_l_channel, (256, 0))
        newIm.save(os.path.join(TARGET, filename))
        i += 1
        if i % 100 == 0:
            print("on iteration", i)
            if i % 1500 == 0:
                break
    with open('filename_to_pic.pkl', 'wb') as f:
        pickle.dump(filename_to_orig_pic, f)