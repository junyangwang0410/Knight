import torch
import json
from clip import clip
import numpy as np
import cv2
import os
from PIL import Image

def video2frames(videofile, savepath):
    vcap = cv2.VideoCapture()
    vcap.open(videofile)
 
    n = 1
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(total_frames / 5)
 
    for i in range(total_frames):
        _, frame = vcap.read()
        if i % frame_interval == 0:
            filename = videofile.split('.')[-1] + '_' + str(n) + '.jpg'

            cv2.imencode('.jpg', frame)[1].tofile(os.path.join(savepath, filename))
            n += 1
            if n == 6:
                break
    assert n == 6
    vcap.release()

device = "cuda:0"
clip_model, preprocess = clip.load("RN50x64", device = device)

with torch.no_grad():
    json_path = "./data/MSRVTT/train_val_videodatainfo.json"
    annotations = json.load(open(json_path,'r'))
    annotations = annotations['sentences']

    captions = []
    for caption in annotations[:140192]:
        captions.append(caption["caption"] + '.')
        
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
    
    torch.save(caption_features, "./feature/MSRVTT/caption_features.pkl")
    captions = np.array(captions)
    np.save("./feature/MSRVTT/captions.npy", captions)
    
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
    np.save("./feature/MSRVTT/nibers.npy", nibers)
    
    json_path = "../Dataset/MSR-VVT/test_videodatainfo.json"
    annotations = json.load(open(json_path,'r'))
    annotations = annotations['sentences']
        
    video_name = []
    video_dict = dict()
    for caption in annotations:
        if video_dict.get(caption["video_id"]):
            continue
        else:
            video_name.append(caption["video_id"])
            video_dict[caption["video_id"]] = 1
        
    video_name.sort()

    for video_info in video_name:
        savepath = './feature/MSRVTT/frames/' + video_info + '/'
        os.mkdir(savepath)
        videofile = "./data/MSRVTT/video/" + video_info + '.mp4'
        video2frames(videofile, savepath)
        
    path = "./feature/MSRVTT/frames/"
    video_names = os.listdir(path)
    
    video_features = []
    for video_name in video_names:
        video_path = path + video_name + '/'
        image_names = os.listdir(video_path)
        video_feature = []
        for image_name in image_names:
            ori_image = Image.open(video_path + image_name)
            image = preprocess(ori_image).unsqueeze(0).to(device)
            image_feature = clip_model.encode_image(image)
            video_feature.append(image_feature)
        video_feature = torch.cat(video_feature)
        video_features.append(video_feature.unsqueeze(0))
    video_features = torch.cat(video_features)

    torch.save(video_features, "./feature/MSRVTT/video_features.pkl")