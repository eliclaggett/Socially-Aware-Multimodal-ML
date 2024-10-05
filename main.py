"""
File Name: main.py
Author: Eli Claggett
Date: Apr 2023

Description:
    Augmented version of Frozen-BiLM model including body pose and facial feature modalities to answer social reasoning questions on the SocialIQ-2 dataset. 
    
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce
import wandb

from datasets import build_mc_dataset, mc_collate_fn
from model import build_model, get_tokenizer
from main import get_args_parser
from util.misc import get_mask, adjust_learning_rate, mask_tokens
from util.metrics import MetricLogger
from transformers import logging

logging.set_verbosity_error()

wandb.login(key='REDACTED')
wandb.init(project='frozen_bilm')

# Linear Probing + Fine Tuning (LP-FT) Configuration

# Step 1) Linear Probing
# epochs = 1
# lr = 3e-4
# args.freeze_notnew=True
# args.freeze_last=False

# Step 2) Fine Tuning
# epochs = 1
# lr = 3e-4
# args.freeze_notnew=False
# args.freeze_last=True


def train_one_epoch(
    model: torch.nn.Module,
    tokenizer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        
        face = batch_dict['face'].to(device)
        face_len = batch_dict["face_len"]
        face_mask = get_mask(face_len, face.size(1)).to(device)
        
        pose = batch_dict['pose'].to(device)
        pose_len = batch_dict["pose_len"]
        pose_mask = get_mask(pose_len, pose.size(1)).to(device)

        text = batch_dict["text"]
        logits_list = []
        for aid in range(len(text)):  # one forward per answer candidate id
            encoded = tokenizer(
                text[aid],
                add_special_tokens=True,
                max_length=args.max_tokens,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            # forward
            output = model(
                video=video,
                video_mask=video_mask,
                face=face,
                face_mask=face_mask,
                pose=pose,
                pose_mask=pose_mask,
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            logits = output["logits"]
            # get logits for the mask token
            delay = args.max_feats if args.use_video else 0
            logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
                encoded["input_ids"] == tokenizer.mask_token_id
            ]
            logits = logits.softmax(-1)
            logits_list.append(logits[:, 0])
        skip_batch=False
        for l in logits_list:
            if (len(l) != args.batch_size):
                skip_batch = True
        if(skip_batch):
            print('skipping batch with wrong number of answers')
            continue
        logits = torch.stack(logits_list, 1)
        gt = batch_dict["answer_id"].to(device)
        if data_loader.dataset.mc > 1:
            pos_logits = logits[torch.arange(len(logits)), gt]
            neg_mask = torch.ones_like(logits)
            neg_mask.scatter_(1, gt.unsqueeze(-1), 0)
            neg_logits = logits[neg_mask[:, :].bool()].view(
                len(logits), data_loader.dataset.mc - 1
            )
            pos_loss = F.binary_cross_entropy(
                pos_logits, torch.ones(len(pos_logits)).to(device)
            )
            neg_logits = neg_logits.view(-1)
            neg_loss = F.binary_cross_entropy(
                neg_logits,
                torch.zeros(len(neg_logits)).to(device),
            )
            loss = (pos_loss + neg_loss) / 2  # balanced BCE
        else:
            loss = F.binary_cross_entropy(logits.squeeze(1), gt.float())

        loss_dict = {"cls_loss": loss}
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )
        if wandb:
            wandb.log({'epoch': epoch+1, 'loss': loss_value, 'lr': optimizer.param_groups[0]["lr"]})

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer,
    data_loader,
    device: torch.device,
    dataset_name,
    args,
    split="test",
    type_map={0: "all"},
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}
    accs = []
    print('Evaluating model')
    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        
        face = batch_dict['face'].to(device)
        face_len = batch_dict['face_len']
        face_mask = get_mask(face_len, face.size(1)).to(device)
        
        pose = batch_dict['pose'].to(device)
        pose_len = batch_dict['pose_len']
        pose_mask = get_mask(pose_len, pose.size(1)).to(device)
        
        text = batch_dict["text"]
        logits_list = []
        for aid in range(len(text)):
            encoded = tokenizer(
                text[aid],
                add_special_tokens=True,
                max_length=args.max_tokens,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            # forward
            output = model(
                video=video,
                video_mask=video_mask,
                face=face,
                face_mask=face_mask,
                pose=pose,
                pose_mask=pose_mask,
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            logits = output["logits"]
            # get logits for the mask token
            delay = args.max_feats if args.use_video else 0
            logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
                encoded["input_ids"] == tokenizer.mask_token_id
            ]
            logits_list.append(logits.softmax(-1)[:, 0])
        if (len(logits_list[-1]) == 0):
            print('logit error, not the right number of questions')
            print(batch_dict['qid'])
            print(logits_list)
            continue
        logits = torch.stack(logits_list, 1)
        if logits.shape[1] == 1:
            preds = logits.round().long().squeeze(1)
        else:
            preds = logits.max(1).indices
        
        qids = batch_dict["qid"]
        types = batch_dict["type"]
        if batch_dict["answer_id"][0].item() != -1:
            answer_id = batch_dict["answer_id"].to(device)
            agreeings = preds == answer_id

            for i, (qid, gt, pred, type) in enumerate(
                zip(qids, answer_id, preds, types)
            ):
                res[qid] = (
                    {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                        "type": int(type),
                    }
                    if type_map is not None and len(type_map) > 1
                    else {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                    }
                )
                res[qid][f"acc"] = agreeings[i].cpu().detach().item()

            dico = {"acc": agreeings.sum() / len(qids)}
            dico_reduced = dist.reduce_dict(dico)
            acc_value = dico_reduced["acc"].item()
            accs.append(acc_value)
            metric_logger.update(acc=np.mean(accs))
        else:
            for i, (qid, pred, type) in enumerate(zip(qids, preds, types)):
                res[str(qid)] = int(pred.cpu().detach().item())


    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})

    if isinstance(next(iter(results.values())), dict):
        acc = sum(int(results[qid][f"acc"]) for qid in results) / len(results)
        if type_map is not None and len(type_map) > 1:
            acc_type = {
                type_map[i]: sum(
                    results[qid][f"acc"] for qid in results if results[qid]["type"] == i
                )
                / len([x for x in results.values() if x["type"] == i])
                for i in type_map
            }
        if dist.is_main_process():
            print(dataset_name)
            print(f"{split} acc: {acc: .2%}")
            if type_map is not None and len(type_map) > 1:
                for x in acc_type:
                    print(f"acc {x}: {acc_type[x]: .2%}")

        return results, acc
    else:
        return results, 0


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    args.n_ans = 2  # Yes and No
    model = build_model(args)
    model.to(device)
    tokenizer = get_tokenizer(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("\nNumber of params:", n_parameters, "\n")

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_test",
            "dataloader_val",
            "dataloader_train",
        ],
    )

    tuples = []
    for dset_name in args.combine_datasets_val:
        if args.n_ans:
            tok_yes = torch.tensor(
                tokenizer(
                    "Yes",
                    add_special_tokens=False,
                    max_length=1,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            tok_no = torch.tensor(
                tokenizer(
                    "No",
                    add_special_tokens=False,
                    max_length=1,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            a2tok = torch.stack([tok_yes, tok_no])
            model.set_answer_embeddings(
                a2tok.to(model.device), freeze_last=args.freeze_last
            )  # initialize answer embedding module
        dataset_test = build_mc_dataset(
            dset_name,
            "val" if (args.eval and not args.test) else "test",
            args,
            tokenizer,
        )
        sampler_test = (
            DistributedSampler(dataset_test, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_test)
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size_val,
            sampler=sampler_test,
            collate_fn=mc_collate_fn,
            num_workers=args.num_workers,
        )

        dataset_val = build_mc_dataset(dset_name, "val", args, tokenizer)
        sampler_val = (
            DistributedSampler(dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_val)
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_val,
            sampler=sampler_val,
            collate_fn=mc_collate_fn,
            num_workers=args.num_workers,
        )

        if not args.eval:
            dataset_train = build_mc_dataset(dset_name, "train", args, tokenizer)
            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                sampler=sampler_train,
                collate_fn=mc_collate_fn,
                num_workers=args.num_workers,
            )
        else:
            dataloader_train = None

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
                dataloader_train=dataloader_train,
            )
        )

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("\nLoading FrozenBiLM checkpoint from", args.load,"\n")
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    for i, item in enumerate(tuples):
        if not args.eval:
            if dist.is_main_process():
                print("Start training")
            start_time = time.time()
            best_epoch = args.start_epoch
            best_acc = 0
            for epoch in range(args.start_epoch, args.epochs):
                if dist.is_main_process():
                    print(f"Starting epoch {epoch}")
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                
                if (args.lp_ft):
                    # Start LP-FT two epochs
                    print(f"Starting linear probing")
                    model.update_settings({'freeze_notnew':True, 'freeze_last':False, 'ft_ln': args.ft_ln, 'freeze_lm':args.freeze_lm})
                    
                    train_stats = train_one_epoch(
                        model=model,
                        tokenizer=tokenizer,
                        data_loader=item.dataloader_train,
                        optimizer=optimizer,
                        device=device,
                        epoch=epoch,
                        args=args,
                        max_norm=args.clip_max_norm,
                    )
                    
                    print(f"Starting fine tuning")
                    model.update_settings({'freeze_notnew':False, 'freeze_last':args.freeze_last, 'ft_ln': args.ft_ln, 'freeze_lm':args.freeze_lm})
                    
                    train_stats = train_one_epoch(
                        model=model,
                        tokenizer=tokenizer,
                        data_loader=item.dataloader_train,
                        optimizer=optimizer,
                        device=device,
                        epoch=epoch+1,
                        args=args,
                        max_norm=args.clip_max_norm,
                    )

                    val_stats = {}
                    for i, item in enumerate(tuples):
                        print(f"Validating {item.dataset_name}")

                        curr_val_stats, acc = evaluate(
                            model=model,
                            tokenizer=tokenizer,
                            data_loader=item.dataloader_val,
                            device=device,
                            dataset_name=item.dataset_name,
                            args=args,
                            split="val",
                            type_map=item.dataloader_val.dataset.type_map,
                        )
                        val_stats[item.dataset_name + "_acc"] = acc
                        if acc > best_acc:
                            best_epoch = epoch
                            best_acc = acc

                            if args.save_dir and dist.is_main_process():
                                checkpoint_path = os.path.join(
                                    args.save_dir, f"best_model.pth"
                                )
                                dist.save_on_master(
                                    {
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "epoch": epoch,
                                        "args": args,
                                    },
                                    checkpoint_path,
                                )
                                json.dump(
                                    curr_val_stats,
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                                json.dump(
                                    {"acc": acc, "ep": epoch},
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "acc_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                    # End LP-FT two epochs
                else:
                    # Start normal fine tuning one epoch
                    train_stats = train_one_epoch(
                        model=model,
                        tokenizer=tokenizer,
                        data_loader=item.dataloader_train,
                        optimizer=optimizer,
                        device=device,
                        epoch=epoch,
                        args=args,
                        max_norm=args.clip_max_norm,
                    )

                    val_stats = {}
                    for i, item in enumerate(tuples):
                        print(f"Validating {item.dataset_name}")

                        curr_val_stats, acc = evaluate(
                            model=model,
                            tokenizer=tokenizer,
                            data_loader=item.dataloader_val,
                            device=device,
                            dataset_name=item.dataset_name,
                            args=args,
                            split="val",
                            type_map=item.dataloader_val.dataset.type_map,
                        )
                        val_stats[item.dataset_name + "_acc"] = acc
                        if acc > best_acc:
                            best_epoch = epoch
                            best_acc = acc

                            if args.save_dir and dist.is_main_process():
                                checkpoint_path = os.path.join(
                                    args.save_dir, f"best_model.pth"
                                )
                                dist.save_on_master(
                                    {
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "epoch": epoch,
                                        "args": args,
                                    },
                                    checkpoint_path,
                                )
                                json.dump(
                                    curr_val_stats,
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                                json.dump(
                                    {"acc": acc, "ep": epoch},
                                    open(
                                        os.path.join(
                                            args.save_dir,
                                            item.dataset_name + "acc_val.json",
                                        ),
                                        "w",
                                    ),
                                )
                    # End normal fine tuning one epoch 

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if args.save_dir and dist.is_main_process():
                    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                    dist.save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            # load best ckpt
            if dist.is_main_process() and args.save_dir:
                print(f"loading best checkpoint from epoch {best_epoch}")
            if args.save_dir:
                torch.distributed.barrier()  # wait all processes
                checkpoint = torch.load(
                    os.path.join(args.save_dir, f"best_model.pth"),
                    map_location="cpu",
                )
                model.load_state_dict(checkpoint["model"], strict=False)

        results, acc = evaluate(
            model=model,
            tokenizer=tokenizer,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            type_map=item.dataloader_test.dataset.type_map,
            split="val" if (args.eval and not args.test) else "test",
        )

        if args.save_dir and dist.is_main_process():
            json.dump(
                results,
                open(
                    os.path.join(
                        args.save_dir,
                        item.dataset_name + "_val.json"
                        if (args.eval and not args.test)
                        else item.dataset_name + "_test.json",
                    ),
                    "w",
                ),
            )
            json.dump(
                {"acc": float(acc)},
                open(
                    os.path.join(
                        args.save_dir,
                        item.dataset_name + "acc_val.json"
                        if (args.eval and not args.test)
                        else item.dataset_name + "acc_test.json",
                    ),
                    "w",
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Frozen training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)

    main(args)
