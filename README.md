# README

[Paper (Accepted by IJCAI 2023)](https://www.ijcai.org/proceedings/2023/481)

## 1. Installing

```bash
$ pip install -r requirements.txt
$ pip install git+https://github.com/openai/CLIP.git
```

## 2. Data Preparation
Downloading the images and videos of each dataset from Web
The data files looks like:
```bash
./data/
  ├──./COCO/
  |   ├──./image/					#images of the test split
  |   ├──captions_val2014.json		#annotation of test split
  |   ├──coco_test.txt				#test split of Karpathy
  ├──./Flickr/
  |   ├──./image/					#images in dataset
  |   ├──dataset_flickr30k.json		#annotation
  ├──./MSRVTT/
  |   ├──./video/					#images in dataset
  |   ├──./frames/					#keyframes
  |   ├──train_val_videodatainfo.json	#annotation
  ├──./MSVD/
  |   ├──./video/					#images in dataset
  |   ├──./frames/					#keyframes
  |   ├──caption.txt				#annotation
  |   ├──train_list.txt				#train split
  |   ├──test_list.txt				#train split
```

After preparing the data, execute the following commands to obtain the data files required to run

```bash
python data_prepare_{dataset name}.py
```
***dataset name = {coco, flickr, msrvtt, msvd}***

## 3. Run

### Image Captioning
```bash
python run_image_captioning.py --dataset {dataset name}
```
***dataset name = {coco, flickr}***

### Video Captioning
```bash
python run_video_captioning.py --dataset {dataset name}
```
***dataset name = {msrvtt, msvd}***

The default save path for checkpoints is **./checkpoint/{dataset name}**, and the default save path for caption flies is **./output/{dataset name}**, where the ***dataset name = {coco, flickr, msrvtt, msvd}***

## 4. Evaluation
We provide the reference results and the results generated as the paper under the **./output/{dataset_name}/**

For example:
```bash
python evalution.py 
--ref ./output/COCO/reference_COCO.json
--gts ./output/COCO/result_COCO.json
```


## 5. Demo
Getting the checkpoint as above operations and put them in ./checkpoint/COCO/ as:
```bash
./checkpoint/
  ├──./COCO/
  |   ├──decoder_coco.pth
  |   ├──map_coco.pth	
```
Then run the ***demo.ipynb***
![](example.jpeg?v=1&type=image)
