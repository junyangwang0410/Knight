import torch
import json
from clip import clip
import numpy as np
from PIL import Image

device = "cuda:0"
clip_model, preprocess = clip.load("RN50x64", device = device)

with torch.no_grad():
    captions = []
    json_path = "./data/COCO/train.json"
    json_labels = json.load(open(json_path,'r'))
    annotations = json_labels
    
    for annotation in annotations[:566720]:
        captions.append(annotation["caption"])
    
    features = []
    index = 0
    batch_size = 256
    while index < len(captions):
        batch_captions = captions[index : index+batch_size]
        clip_captions = clip.tokenize(batch_captions).to(device)
        clip_features = clip_model.encode_text(clip_captions)
        features.append(clip_features)
        index += batch_size
        
    caption_features = torch.cat(features)
    
    torch.save(caption_features, "./feature/COCO/caption_features.pkl")
    captions = np.array(captions)
    np.save("./feature/COCO/captions.npy", captions)
    
    caption_features = caption_features / caption_features.norm(dim = -1, keepdim = True)

    nibers = []
    for i in range(caption_features.shape[0]):
        caption_feature = caption_features[i].unsqueeze(0)
        similarity = caption_feature @ caption_features.T
        similarity[0][i] = 0
        niber = []
        for j in range(5):
            _, max_id = torch.max(similarity, dim = 1)
            niber.append(max_id.item())
            similarity[0][max_id.item()] = 0
            
        nibers.append(niber)

    nibers = np.array(nibers)
    np.save("./feature/COCO/nibers.npy", nibers)

    json_path = "./data/COCO/captions_val2014.json"
    json_labels = json.load(open(json_path,'r'))
    annotations = json_labels["annotations"]
    images = json_labels["images"]
    images_path = "./data/COCO/image/"

    image_dict = dict()
    for image in images:
        image_dict[image["file_name"]] = image["id"]

    with open("./data/COCO/coco_test.txt") as image_names_data:
        image_names = image_names_data.readlines()

    image_features = []
    for image_info in image_names:
        image_file = image_info.split('\n')[0]
        image_id = image_dict[image_file]
        image_path = images_path + image_file
        ori_image = Image.open(image_path)
        image = preprocess(ori_image).unsqueeze(0).to(device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
        
    image_features = torch.cat(image_features)
    torch.save(image_features, "./feature/COCO/image_features.pkl")