# Load detections for backwards projection 
import pandas as pd
import numpy as np

def load_detections(path):
    """Load a csv file of bounding box detections
    CSV takes the format xmin, ymin, xmax, ymax, image_path, label. The bounding box corners are in the image coordinate system,
    the image_path is the expected to be the full path, and the label is the character label for each box. One detection per row in the csv.
    
    Args:
        path: path on local disk
    Returns:
        boxes: a pandas dataframe of detections
    """
    
    boxes = pd.read_csv(path)
    if not all([x in ["image_path","xmin","ymin","xmax","ymax","image_path","label"] for x in boxes.columns]):
        raise IOError("{} is expected to be a .csv with columns, xmin, ymin, xmax, ymax, image_path, label for each detection")
    
    boxes.to_dict()
    
    return boxes