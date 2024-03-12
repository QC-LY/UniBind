import torch
import models.PointBind_models as models
from imagebind.imagebind_model import ModalityType
import argparse
import json
import pickle
from tqdm import tqdm
from data.process_data import EvalVisionDataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import logging
import random
import numpy as np
import os
import pickle
from utils.utils import set_env, gen_label, loss_fun, load_centre_embeddings
from model import UniBind

logger = logging.getLogger(__name__)

def gen_visual_embeddings(model, data_loader):
    all_embeddings = []
    all_visual_labels = []
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            all_visual_labels.extend(batch['labels'])
            embeddings = model.encode_vision(batch['inputs'])
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_visual_labels

def evaluate(args, model, val_data_loader, device):
    model.eval()
    logger.info('Start to load centre embeddings!')
    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    logger.info('centre embedding load done!')
    logger.info('---------------------------------')
    logger.info('Start to generate visual embeddings!')
    visual_embeddings, visual_labels = gen_visual_embeddings(model, val_data_loader)
    logger.info('visual embedding generation done!')
    logger.info('---------------------------------')
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)
    visual_embeddings /= visual_embeddings.norm(dim=-1, keepdim=True)
    logic = (visual_embeddings.to(device) @ centre_embeddings.to(device).t()).softmax(dim=-1)
    acc = 0.0
    for i in range(logic.shape[0]):
        _, index = logic[i].topk(1)
        if visual_labels[i] == centre_labels[int(index[0])]:
            acc = acc + 1
    return acc/logic.shape[0]


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--test_dataset_dir", type=str, default='', required=True)
    parser.add_argument("--test_data_path", type=str, default='', required=True)
    parser.add_argument("--centre_embeddings_path", type=str, default='', required=True)
    parser.add_argument("--output_dir", type=str, default='', required=True)
    parser.add_argument("--pretrain_weights", type=str, default='', required=True)
    parser.add_argument("--modality", type=str, default='vision', required=True)
    parser.add_argument("--val_batch_size", type=int, default=8, required=True)
    parser.add_argument("--num_workers", type=int, default=0, required=True)
    parser.add_argument("--seed", type=int, default=1234, required=True)

    args = parser.parse_args()
    log_name = args.modality + '_infer'
    set_env(args, log_name)

    val_data = EvalVisionDataset(args, device, infer_type="test")
    val_sampler = SequentialSampler(val_data)
    val_data_reader = DataLoader(dataset=val_data, sampler=val_sampler, num_workers=args.num_workers,
                                batch_size=args.val_batch_size, collate_fn=val_data.Collector, drop_last=False)
    
    model = UniBind(args)
    model.to(device)
    acc = evaluate(args, model, val_data_reader, device)
    logger.info(f"top 1 Acc: {acc}")