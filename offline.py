from PIL import Image
from pathlib import Path
import numpy as np
from Feature_Extractor import FeatureExtractor

def train():
    if __name__=="__main_":

            fe = FeatureExtractor()

    for img_path in sorted(Path("C:\Thesis\static\Datasets").glob ("*.jpg")): 
            print(img_path)
            fe = FeatureExtractor()

            Feature=fe.extract(img=Image.open(img_path))

            feature_path = Path("C:\Thesis\static\Feature") / (img_path.stem + ".npy") 
            print(feature_path)

            np.save(feature_path, Feature)