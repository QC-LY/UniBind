import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import random
from utils.data_transform import (
    load_and_transform_vision_data, load_and_transform_video_data, load_and_transform_audio_data,
    load_and_transform_text, load_and_transform_thermal_data, load_and_transform_point_data
)
from imagebind.imagebind_model import ModalityType
import os

class TrainDataset(Dataset):
    def __init__(self, args, device):
        self.dataset_dir = args.train_dataset_dir
        self.data_path = args.train_data_path
        self.device = device
        self.modality = args.modality
        self.visual_data = []
        self.descriptions = []
        with open(self.data_path, 'r') as fin:
            data_list = json.load(fin)
            random.shuffle(data_list)
            for line in data_list[:100]:
                visual_data_path = os.path.join(self.dataset_dir, line['data'])
                self.visual_data.append(visual_data_path)
                self.descriptions.append(line['description'])

    def __len__(self):
        return len(self.visual_data)

    def __getitem__(self, index):
        visual_data = self.visual_data[index]
        description = self.descriptions[index]
        data_sample = (visual_data, description)
        return data_sample
    
    def Collector(self, batch):
        visual_data_list = []
        description_list = []
        for i, example in enumerate(batch):
            visual_data_list.append(example[0])
            description_list.append(example[1])
        if self.modality == 'image':
            inputs = {
                ModalityType.VISION: load_and_transform_vision_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        if self.modality == 'video':
            inputs = {
                ModalityType.VISION: load_and_transform_video_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        if self.modality == 'audio':
            inputs = {
                ModalityType.AUDIO: load_and_transform_audio_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        if self.modality == 'thermal':
            inputs = {
                ModalityType.THERMAL: load_and_transform_thermal_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        if self.modality == 'event':
            inputs = {
                ModalityType.VISION: load_and_transform_vision_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        if self.modality == 'point':
            inputs = {
                ModalityType.POINT: load_and_transform_point_data(visual_data_list, self.device),
                ModalityType.TEXT: load_and_transform_text(description_list, self.device)
            }
        processed_batch = {
            'inputs': inputs
        }

        return processed_batch


class EvalVisionDataset(Dataset):
    def __init__(self, args, device, infer_type="eval"):
        if infer_type == "eval":
            self.data_path = args.eval_data_path
            self.dataset_dir = args.eval_dataset_dir
        else:
            if infer_type == "test":
                self.data_path = args.test_data_path
                self.dataset_dir = args.test_dataset_dir
        self.device = device
        self.modality = args.modality
        self.visual_data = []
        self.labels = []
        with open(self.data_path, 'r') as fin:
            data_list = json.load(fin)
            for line in data_list:
                visual_data_path = os.path.join(self.dataset_dir, line['data'])
                self.visual_data.append(visual_data_path)
                self.labels.append(line['label'])

    def __len__(self):
        return len(self.visual_data)

    def __getitem__(self, index):
        visual_data = self.visual_data[index]
        label = self.labels[index]
        data_sample = (visual_data, label)
        return data_sample
    
    def Collector(self, batch):
        visual_data_list = []
        label_list = []
        for _, example in enumerate(batch):
            visual_data_list.append(example[0])
            label_list.append(example[1])
        if self.modality == "image":
            inputs = {
                ModalityType.VISION: load_and_transform_vision_data(visual_data_list, self.device)
            }
        if self.modality == "video":
            inputs = {
                ModalityType.VISION: load_and_transform_video_data(visual_data_list, self.device)
            }
        if self.modality == "audio":
            inputs = {
                ModalityType.AUDIO: load_and_transform_audio_data(visual_data_list, self.device)
            }
        if self.modality == "thermal":
            inputs = {
                ModalityType.THERMAL: load_and_transform_thermal_data(visual_data_list, self.device)
            }
        if self.modality == "event":
            inputs = {
                ModalityType.VISION: load_and_transform_vision_data(visual_data_list, self.device)
            }
        if self.modality == "point":
            inputs = {
                ModalityType.POINT: load_and_transform_point_data(visual_data_list, self.device)
            }
        processed_batch = {
            'inputs': inputs,
            'labels': label_list
        }

        return processed_batch
    
