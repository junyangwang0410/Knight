import torch
import json
from clip import clip
import numpy as np
from PIL import Image

device = "cuda:0"
clip_model, preprocess = clip.load("RN50x64", device = device)

with torch.no_grad():
    num = 0
    captions = []
    references = []
    json_path = "./data/Flickr/dataset_flickr30k.json"
    json_labels = json.load(open(json_path,'r'))
    annotations = json_labels['images']
    
    for annotation in annotations:
        if annotation['split'] == 'train':
            for caption in annotation['sentences']:
                captions.append(caption["raw"].lower()[:300])
                
        elif annotation['split'] == 'test':
            num += 1
            reference = []
            for caption in annotation['sentences']:
                reference.append(caption["raw"].lower()[:300])
            references.append(reference)
        
    captions = captions[:144992]
    
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
    
    torch.save(caption_features, "./feature/Flickr/caption_features.pkl")
    captions = np.array(captions)
    np.save("./feature/Flickr/captions.npy", captions)

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
    np.save("./feature/Flickr/nibers_resnet.npy", nibers)
    
    image_names = []
    for annotation in annotations:
        if annotation['split'] == 'test':
            image_names.append(annotation['filename'])

    image_path = "./data/Flickr/image/"
    image_features = []
    for image_name in image_names:
        ori_image = Image.open(image_path + image_name)
        image = preprocess(ori_image).unsqueeze(0).to(device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
        
    image_features = torch.cat(image_features)
    torch.save(image_features, "./feature/Flickr/image_features_resnet.pkl")