from model import Mlp, caption_generation
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_point", type=int, default=468201)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0000001)
    parser.add_argument("--mlp_lr", type=float, default=10.0)
    parser.add_argument("--prefix", type=str, default="prefix prefix prefix prefix prefix:")
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--print_frequency", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/")
    parser.add_argument("--output_dir", type=str, default="./output/")
    args = parser.parse_args()
    return args

def main(args):
    device = args.device
    
    if args.dataset == 'coco':
        dataset_name = 'COCO'
    elif args.dataset == 'flickr':
        dataset_name = 'Flickr'
    else:
        print('Please input correct dataset!')
        assert args.dataset == 'coco' or args.dataset == 'flickr'
        
    nibers = np.load("./feature/" + dataset_name + "/nibers.npy")
    nibers = nibers.tolist()
    captions = np.load("./feature/" + dataset_name + "/captions.npy")
    captions = captions.tolist()
    caption_features = torch.load("./feature/" + dataset_name + "/caption_features.pkl").to(device)
    image_features = torch.load("./feature/" + dataset_name + "/image_features.pkl").to(device)
    caption_features_norm = caption_features / caption_features.norm(dim = -1, keepdim = True)
    
    GPT_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    GPT_tokenizer.pad_token = GPT_tokenizer.eos_token
    GPT_tokenizer.cls_token = GPT_tokenizer.eos_token
    GPT_tokenizer.sep_token = GPT_tokenizer.eos_token
    GPT_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    GPT_model.to(device)
    Map = Mlp()
    Map.to(device)
    
    batch_size = args.batch_size
    prefix = args.prefix
    optimizer = torch.optim.AdamW([{"params": GPT_model.parameters()}, 
                                   {"params": Map.parameters(), "lr": args.lr*args.mlp_lr}], 
                                  lr = args.lr, weight_decay = args.weight_decay)
    seed = args.seed
    torch.manual_seed(seed)
    
    for epoch in range(args.epoch):
        index = 0
        l = 0
        n = 0
        GPT_model.train()
        Map.train()

        while index < len(captions):
            batch_niber = nibers[index : index+batch_size]
            batch_caption = captions[index : index+batch_size]
            batch_caption_feature = []
            for niber in batch_niber:
                feature = []
                for i in niber:
                    noise = torch.randn(1024).to(device)
                    this_feature = caption_features[i] + args.noise * noise
                    seed += torch.randint(1, 10, (1,)).item()
                    seed %= args.seed_point
                    torch.manual_seed(seed)
                    feature.append(this_feature.unsqueeze(0))
                
                feature = torch.cat(feature)
                batch_caption_feature.append(feature.unsqueeze(0))

            batch_caption_feature = torch.cat(batch_caption_feature).to(device)
            batch_caption_feature = Map(batch_caption_feature)
            
            for i in range(len(batch_caption)):
                batch_caption[i] = prefix + batch_caption[i].split('.')[0] + '.'
                
            token = GPT_tokenizer(batch_caption, return_tensors="pt", padding = True, truncation = True, max_length = args.max_length).to(device)
            output = GPT_model(**token, labels = token["input_ids"], prefix = batch_caption_feature)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()
            n += 1
            
            with torch.no_grad():  
                index += batch_size
                
                if index % (batch_size * args.print_frequency) == 0:
                    torch.cuda.empty_cache()
                    print("[ Epoch:", epoch, "]", index, '/', len(captions), "loss =", l/(n + 0.000001))
                    l = 0
                    n = 0
    
        torch.cuda.empty_cache()
        torch.save(GPT_model.state_dict(), args.checkpoint_dir + dataset_name + f"/decoder{epoch}.pth")
        torch.save(Map.state_dict(), args.checkpoint_dir + dataset_name + f"/mlp{epoch}.pth")

        with torch.no_grad():
            gts_dict = dict()
            num = 0
            
            GPT_model.eval()
            Map.eval()

            for i in tqdm(range(len(image_features))):
                image_feature = image_features[i].unsqueeze(0)
                image_feature = image_feature / image_feature.norm(dim = -1, keepdim = True)
                similarity = image_feature @ caption_features_norm.T

                niber = []
                for k in range(5):
                    _, max_id = torch.max(similarity, dim = 1)
                    niber.append(caption_features[max_id.item()].unsqueeze(0))
                    similarity[0][max_id.item()] = 0
                    
                niber = torch.cat(niber).unsqueeze(0)
                niber = Map(niber.float())
                candidate = caption_generation(niber, GPT_model, GPT_tokenizer, device)
                gts_dict[num] = [candidate.replace(",", " ,")]
                num += 1
                
            gts = open(args.output_dir + dataset_name + f"/result_{epoch}.json", "w")
            json.dump(gts_dict, gts)
            gts.close()
            
if __name__ == "__main__":
    args = get_args()
    main(args)