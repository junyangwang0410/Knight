import random
import torch
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

with open("../Dataset/MSVD/caption.txt") as caption_data:
    captions_info = caption_data.readlines()

caption_dict = dict()

for caption_info in captions_info:
    video_name = caption_info.split(' ')[0]
    caption = caption_info[len(video_name)+1:].split('\n')[0]
    caption_dict.setdefault(video_name, []).append(caption)

with open("../Dataset/MSVD/train_list.txt") as train_data:
    train_names = train_data.readlines()

captions = []
for train_name in train_names:
    train_name = train_name.split('\n')[0]
    for caption in caption_dict[train_name]:
        captions.append(caption + '.')

random.shuffle(captions)

with open("../Dataset/MSVD/test_list.txt") as test_data:
    test_names = test_data.readlines()

with torch.no_grad():
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
    
    torch.save(caption_features, "./feature/MSVD/caption_features.pkl")
    captions = np.array(captions)
    np.save("./feature/MSVD/captions.npy", captions)

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
    np.save("./feature/MSVD/nibers.npy", nibers)
    
    with open("./data/MSVD/test_list.txt") as video_name_data:
        video_name = video_name_data.readlines()

    for video_info in video_name:
        video_info = video_info.split('\n')[0]
        savepath = './feature/MSVD/frames/' + video_info + '/'
        os.mkdir(savepath)
        videofile = "./data/MSVD/video/" + video_info + '.avi'
        video2frames(videofile, savepath)
        
    path = "./feature/MSVD/frames/"
    with open("../Dataset/MSVD/test_list.txt") as video_name_data:
        video_names = video_name_data.readlines()

    video_features = []
    for video_name in video_names:
        video_name = video_name.split('\n')[0]
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

    torch.save(video_features, "./feature/MSVD/video_features.pkl")