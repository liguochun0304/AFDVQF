# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: train.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import json
import os
from datetime import datetime

import swanlab
import torch
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, CLIPProcessor, get_linear_schedule_with_warmup
from transformers import BertConfig
from transformers import BertTokenizer
from dataloader import MultimodalNERDataset, collate_fn
from model import MultimodalNER
import os
import random
import numpy as np
import torch



script_dir = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed=42):
    random.seed(seed)                    # Python 随机种子
    np.random.seed(seed)                 # numpy 随机种子
    torch.manual_seed(seed)              # CPU torch 随机种子
    torch.cuda.manual_seed(seed)         # GPU 随机种子
    torch.cuda.manual_seed_all(seed)     # 多 GPU 情况

    # 保证 CUDA 可复现（但可能会略微降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
def save_model_checkpoint(model, optimizer, scheduler, config, save_dir, epoch, best_metric):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump({"epoch": epoch, "best_f1": best_metric}, f, indent=2)


def load_model_checkpoint(model, optimizer, scheduler, load_dir):
    model.load_state_dict(torch.load(os.path.join(load_dir, "model.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(load_dir, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(load_dir, "scheduler.pt")))

    with open(os.path.join(load_dir, "training_state.json")) as f:
        state = json.load(f)

    return state["epoch"], state["best_f1"]


def evaluate(model, val_loader, device, id2label):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            # 预测的标签 id 序列
            preds = model(input_ids, attention_mask, image_tensor)

            for p_ids, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = mask.sum().item()
                # 截取有效 token，映射成标签字符串
                pred_labels = [id2label[i] for i in p_ids[:valid_len]]
                true_labels = [id2label[i.item()] for i in l_ids[:valid_len]]

                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    # 实体级别评估
    f1 = seq_f1_score(all_labels, all_preds)
    report = seq_classification_report(all_labels, all_preds, zero_division=0, digits=4, output_dict=True)
    return f1, report


def train(config):
    print("train config:", config)
    swanlab_name = f"{datetime.now().strftime('%Y-%m-%d')}_train-{config.dataset_name}_ex{str(config.ex_nums)}"
    print("train pth save name and swanlab name is", swanlab_name)
    # 初始化实验ex_project
    swanlab.init(
        project=config.ex_project,
        name=f"{swanlab_name}_{config.ex_name}",
        config={
            "fin_tuning_lr": config.fin_tuning_lr,
            "clip_lr": config.clip_lr,
            "downs_en_lr": config.downs_en_lr,
            "weight_decay": config.weight_decay_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs
        },
        dir=os.path.join(script_dir, "swanlog")
    )

    save_dir = os.path.join(script_dir, "save_models", f"{swanlab_name}_{config.ex_name}")

    device = torch.device(config.device)
    if config.text_encoder == "bert-base-uncased" or config.text_encoder == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(os.path.join(script_dir, config.text_encoder))
    else:
        tokenizer = RobertaTokenizer.from_pretrained(os.path.join(script_dir, config.text_encoder))
    processor = CLIPProcessor.from_pretrained(os.path.join(script_dir, config.image_encoder))

    train_dataset = MultimodalNERDataset(config.dataset_name, tokenizer, processor, max_length=config.max_len,
                                         mode="train")
    val_dataset = MultimodalNERDataset(config.dataset_name, tokenizer, processor, max_length=config.max_len,
                                       mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    #
    # dataset = MultimodalNERDataset(
    #     data_path="data/twitter2017",
    #     tokenizer=tokenizer,
    #     processor=clip_processor,
    #     max_length=128,
    #     mode="train"
    # )
    #
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    #
    #
    # train_dataset = MultimodalNERDataset(data_path=config.dataset_name, mode="train")
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    #
    # val_dataset = MultimodalNERDataset(data_path=config.dataset_name, mode="valid")
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    model = MultimodalNER(num_labels=len(train_dataset.id2label), text_encoder_path=config.text_encoder,
                          use_image=config.use_image,
                          fusion_type=config.fusion_type,
                          use_coattention=config.use_coattention,
                          ).to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    param_optimizer = list(model.named_parameters())
    # 模块分类
    param_roberta = [(n, p) for n, p in param_optimizer if "text_encoder" in n]
    param_clip = [(n, p) for n, p in param_optimizer if "image_encoder" in n]
    param_downstream = [(n, p) for n, p in param_optimizer if "text_encoder" not in n and "image_encoder" not in n]

    # 构造 optimizer_grouped_parameters
    optimizer_grouped_parameters = [
        # Roberta
        {
            "params": [p for n, p in param_roberta if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay_rate,
            "lr": config.fin_tuning_lr,
        },
        {
            "params": [p for n, p in param_roberta if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config.fin_tuning_lr,
        },
        # CLIP
        {
            "params": [p for n, p in param_clip if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay_rate,
            "lr": config.clip_lr,
        },
        {
            "params": [p for n, p in param_clip if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config.clip_lr,
        },
        # 下游模块
        {
            "params": [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay_rate,
            "lr": config.downs_en_lr,
        },
        {
            "params": [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config.downs_en_lr,
        },
    ]
    # optimizer = torch.optim.AdamW(grouped_params, lr=config.downs_en_lr, weight_decay=config.weight_decay_rate)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    t_total = len(train_loader) // config.gradient_accumulation_steps * config.epochs
    warmup_steps = int(t_total * config.warmup_prop)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)
        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            loss = model(input_ids, attention_mask, image_tensor, labels)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            # 每个 step 的 gradient norm 和 learning rate 记录（在 optimizer.step 之前）
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 计算 grad_norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                #  clip 梯度，优化器步进
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                #  记录 learning_rate（只记录第一组）
                current_lr = scheduler.get_last_lr()[0]

                #  SwanLab 记录
                swanlab.log({
                    "train/grad_norm": total_norm,
                    "train/learning_rate": current_lr,
                    "step": epoch * len(train_loader) + step
                })

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(train_loader)
        swanlab.log({"train/loss": avg_loss})
        print(f"\n✅ Epoch {epoch} Train Loss: {avg_loss:.4f}")

        f1, report = evaluate(model, val_loader, device, train_dataset.id2label)
        print(
            f"🎯Epoch {epoch} Eval F1: {f1:.4f} precision: {report['weighted avg']['precision']:.4f} recall: {report['weighted avg']['recall']:.4f}")

        swanlab.log({
            "eval/f1": f1,
            "eval/precision": report["weighted avg"]["precision"],
            "eval/recall": report["weighted avg"]["recall"],
            "epoch": epoch
        })

        # Early Stop
        if f1 > best_f1 + config.patience:
            best_f1 = f1
            patience_counter = 0
            save_model_checkpoint(model, optimizer, scheduler, config, save_dir, epoch, best_f1)
            print(f"✅ Model saved to {save_dir}")
        else:
            patience_counter += 1
            print(f"📉 No improvement, patience {patience_counter}/{config.patience_num}")
            if epoch >= config.min_epoch_num and patience_counter >= config.patience_num:
                print("⛔️ Early stopping triggered.")
                break


if __name__ == "__main__":
    from config import get_config

    set_seed(42)
    config = get_config()
    train(config)
