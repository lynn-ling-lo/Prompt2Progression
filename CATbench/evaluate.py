import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import os
import clip
from vbench.utils import load_video
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)


import argparse



def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='CATBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        default='./Prompts/age.txt',
        
    )


    args = parser.parse_args()
    return args

def load_prompts_mp(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = []
        line = line.strip()
        prompts =  line.split(';')
        for prompt in prompts:
            prompt = prompt.strip()
            if len(prompt) != 0:
                l.append(prompt)
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    #print(prompt_list)
    return prompt_list

class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.device = device

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(self.device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity

def clip_directional_score(video_list, prompt_list,args):
    device = torch.device("cuda")
    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder, device)
    coarse_scores = []
    fine_score_mean = []


    for name, prompt in zip(video_list, prompt_list):
        video_path = os.path.join(args.videos_path, name)
        images = load_video(video_path)
        # coarse score
        first_frame = images[0]
        last_frame = images[-1]
        first_prompt = prompt[0]
        last_prompt = prompt[-1]
        sim_score = dir_similarity(first_frame, last_frame, first_prompt, last_prompt)
        coarse_scores.append(sim_score.detach().cpu().numpy())
        # fine score
        consecutive_scores = []
        for i in range(1,len(images)):
            sim_score = dir_similarity(images[i-1], images[i], first_prompt, last_prompt)
            consecutive_scores.append(sim_score.detach().cpu().numpy())
        mean = np.mean(consecutive_scores, axis=0)
        fine_score_mean.append(mean)
    return np.mean(coarse_scores, axis=0), np.mean(fine_score_mean, axis=0)


def main():
    args = parse_args()

    video_list = os.listdir(args.videos_path)
    prompt_list = load_prompts_mp(args.prompt_file)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    coarse_score, fine_mean = clip_directional_score(video_list, prompt_list,args)
    print('coarse clip directional score: ',coarse_score)
    print('fine clip directional score: ',fine_mean)
    

if __name__ == "__main__":
    main()
