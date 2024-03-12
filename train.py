import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import models.PointBind_models as models
from imagebind.imagebind_model import ModalityType
import argparse
import json
import pickle
import logging
import os
from tqdm import tqdm
from data.process_data import TrainDataset, EvalVisionDataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
import random
import numpy as np
from model import UniBind
from utils.utils import set_env, gen_label, loss_fun, load_centre_embeddings

logger = logging.getLogger(__name__)

def gen_visual_embeddings(model, data_loader):
    all_embeddings = []
    all_visual_labels = []
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            all_visual_labels.extend(batch['labels'])
            embeddings = model.encode_vision_with_mlp(batch['inputs'])
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_visual_labels


def evaluate(args, model, val_data_loader, device):
    model.eval()
    logger.info('Start to generate visual embeddings!')
    visual_embeddings, visual_labels = gen_visual_embeddings(model, val_data_loader)
    logger.info('visual embedding generation done!')
    logger.info('---------------------------------')
    logger.info('Start to load centre embeddings!')
    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    logger.info('centre embedding load done!')
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

def train(args, model, train_dataloader, val_dataloader, device):
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    t_total = train_dataloader.dataset.__len__() // real_batch_size * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer], 'weight_decay': 0.01}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss = 0.0
    best_acc = 0.0
    writer = SummaryWriter(log_dir=(args.output_dir + '/tb_loss'))
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        epoch_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            model.train()
            text_embeddings, vision_embeddings = model(batch['inputs'])
            logic = vision_embeddings @ text_embeddings.t()
            labels = gen_label(logic, device)
            loss = loss_fun(logic, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                logger.info(
                    'Epoch: {}, Step: {}, Loss: {:.8f}, lr: {:.6f}'.format(epoch, global_step, (tr_loss / global_step),
                                                                           optimizer.param_groups[0]["lr"]))
                writer.add_scalar(tag="cl_loss", scalar_value= tr_loss / global_step, global_step=step+epoch_steps*epoch)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if global_step % args.eval_steps == 0:
                    logger.info('Start eval!')
                    acc = evaluate(args, model, val_dataloader, device)
                    logger.info('Dev acc: {0}'.format(acc))
                    if acc >= best_acc:
                        best_acc = acc
                        torch.save(
                            model.state_dict(), 
                            os.path.join(args.output_dir, f"{args.modality}_model_best.pt")
                        )
    writer.close()
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.modality}_model_{args.num_train_epochs}.pt"))

def test(args, val_dataloader, device):
    model = UniBind(args)
    model.to(device)
    ckpt_path = os.path.join(args.output_dir, f"{args.modality}_model_best.pt")
    model.load_state_dict(torch.load(ckpt_path))
    acc = evaluate(args, model, val_dataloader, device)
    return acc


def main():
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_dir", type=str, default='', required=True)
    parser.add_argument("--eval_dataset_dir", type=str, default='', required=True)
    parser.add_argument("--test_dataset_dir", type=str, default='', required=True)
    parser.add_argument("--train_data_path", type=str, default='', required=True)
    parser.add_argument("--eval_data_path", type=str, default='', required=True)
    parser.add_argument("--test_data_path", type=str, default='', required=True)
    parser.add_argument("--centre_embeddings_path", type=str, default='', required=True)
    parser.add_argument("--pretrain_weights", type=str, default='', required=True)
    parser.add_argument("--output_dir", type=str, default='', required=True)
    parser.add_argument("--modality", type=str, default='vision', required=True)
    parser.add_argument("--train_batch_size", type=int, default=8, required=True)
    parser.add_argument("--val_batch_size", type=int, default=8, required=True)
    parser.add_argument("--num_workers", type=int, default=0, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=True, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float, required=True, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--eval_steps", type=int, default=10, required=True, help="eval model every X updates steps.")
    parser.add_argument("--seed", type=int, default=1234, required=True, help="random seed for initialization")

    args = parser.parse_args()
    log_name = args.modality + '_finetune'
    set_env(args, log_name)
    model = UniBind(args)
    model.to(device)
    logger.info("Loading training set.")
    train_data = TrainDataset(args, device)
    train_sampler = RandomSampler(train_data)
    train_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=args.num_workers,
                              batch_size=args.train_batch_size, collate_fn=train_data.Collector)
    eval_data = EvalVisionDataset(args, device, infer_type="eval")
    eval_sampler = SequentialSampler(eval_data)
    eval_reader = DataLoader(dataset=eval_data, sampler=eval_sampler, num_workers=args.num_workers,
                              batch_size=args.val_batch_size, collate_fn=eval_data.Collector, drop_last=False)
    test_data = EvalVisionDataset(args, device, infer_type="test")
    test_sampler = SequentialSampler(test_data)
    test_reader = DataLoader(dataset=test_data, sampler=test_sampler, num_workers=args.num_workers,
                              batch_size=args.val_batch_size, collate_fn=eval_data.Collector, drop_last=False)
    
    train(args, model, train_reader, eval_reader, device)

    test_acc = test(args, test_reader, device)
    logger.info(f"Acc on test set: {test_acc}")


if __name__ == "__main__":
    main()