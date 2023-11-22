import os
import cv2
import numpy as np

from skimage.transform import resize

TRAIN_PATH_DATASET = "./data/train/"
TEST_PATH_DATASET = "./data/test/"

HIPPO_DIR = "hippo/"
RHINO_DIR = "rhino/"

IMG_SHAPE = (224, 224)

def load_dataset(test=False, transform=False):
    path_dataset = TEST_PATH_DATASET if test else TRAIN_PATH_DATASET
    
    hippo_path = os.path.join(path_dataset, HIPPO_DIR)
    rhino_path = os.path.join(path_dataset, RHINO_DIR)
    
    hippo_images = []
    rhino_images = []
    
    # Load Hippo images
    for filename in os.listdir(hippo_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(hippo_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if transform:
                img = resize(img, IMG_SHAPE, anti_aliasing=True)
                
            hippo_images.append(img)

    # Load Rhino images
    for filename in os.listdir(rhino_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(rhino_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if transform:
                img = resize(img, IMG_SHAPE, anti_aliasing=True)
                
            rhino_images.append(img)

    # Create corresponding labels (0 for hippo, 1 for rhino)
    hippo_labels = np.zeros(len(hippo_images))
    rhino_labels = np.ones(len(rhino_images))

    # Combine images and labels
    images = np.concatenate([hippo_images, rhino_images], axis=0)
    labels = np.concatenate([hippo_labels, rhino_labels], axis=0)

    return images, labels


# Fonction pour sauvegarder les images à partir de array
def save_images(images, labels, test=False):
    path_dataset = TEST_PATH_DATASET if test else TRAIN_PATH_DATASET

    n_hippo = 1
    n_rhino = 1
    
    # Convertir et sauvegarder chaque image
    for i, image in enumerate(images):
        img_dir = HIPPO_DIR if labels[i] == 0 else RHINO_DIR
        n_img = n_hippo if labels[i] == 0 else n_rhino
        
        save_path = os.path.join(path_dataset, img_dir)

        # Créer le chemin complet pour sauvegarder l'image
        image_name = f"{n_img}.jpg"
        image_path = os.path.join(save_path, image_name)

        if labels[i] == 0:
            n_hippo += 1
        else:
            n_rhino += 1
            
        image = image.reshape(*IMG_SHAPE, 1)

        image = (image * 255).astype(np.uint8)
        cv2.imwrite(image_path, image)