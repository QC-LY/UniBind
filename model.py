import torch
from torch import nn
import torch.nn.functional as F
import models.PointBind_models as models
from imagebind.imagebind_model import ModalityType
import numpy as np

class UniBind(nn.Module):
    def __init__(self, args):
        super(UniBind, self).__init__()
        self.modality = args.modality
        self.backbone = models.PointBind_I2PMAE()
        state_dict = torch.load(args.pretrain_weights, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        if self.modality == "image":
            self.mlp_for_image = nn.Linear(1024,1024)
        if self.modality == "video":
            self.mlp_for_video = nn.Linear(1024,1024)
        if self.modality == "audio":
            self.mlp_for_audio = nn.Linear(1024,1024)
        if self.modality == "thermal":
            self.mlp_for_thermal = nn.Linear(1024,1024)
        if self.modality == "point":
            self.mlp_for_point = nn.Linear(1024,1024)
        if self.modality == "event":
            self.mlp_for_event = nn.Linear(1024,1024)

    def forward(self, inputs):
        if self.modality == "image":
            outputs = self.backbone.bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        if self.modality == "video":
            outputs = self.backbone.bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        if self.modality == "audio":
            outputs = self.backbone.bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        if self.modality == "thermal":
            outputs = self.backbone.bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        if self.modality == "event":
            outputs = self.backbone.bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_features = self.backbone.bind.modality_head_point(pc_features)
            pc_features = self.backbone.bind.modality_postprocessor_point(pc_features)
            outputs = self.backbone.bind({ModalityType.TEXT: inputs['text']})
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_point(pc_embeddings)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings, vision_embeddings
    
    def encode_vision(self, inputs):
        if self.modality == "image":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "video":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "audio":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.AUDIO]
        if self.modality == "thermal":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.THERMAL]
        if self.modality == "event":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return vision_embeddings
    
    def encode_vision_with_mlp(self, inputs):
        if self.modality == "image":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        if self.modality == "video":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        if self.modality == "audio":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        if self.modality == "thermal":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        if self.modality == "event":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.modality_postprocessor_point(pc_embeddings)
            vision_embeddings = self.mlp_for_point(pc_embeddings)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return vision_embeddings
    

    def encode_text(self, inputs):
        text_embeddings = self.backbone.bind(inputs)[ModalityType.TEXT]
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings
