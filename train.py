# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: train.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import os, json, torch, argparse
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPProcessor, get_linear_schedule_with_warmup
from model import MultimodalNER
from dataloader import MultimodalNERDataset, collate_fn
from sklearn.metrics import classification_report
from tqdm import tqdm
import swanlab


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


def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            pred = model(input_ids, attention_mask, image_tensor)
            for p, l, m in zip(pred, labels, attention_mask):
                valid_len = m.sum().item()
                all_preds.extend(p[:valid_len])
                all_labels.extend(l[:valid_len].cpu().tolist())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return report["weighted avg"]["f1-score"], report


def train(config):
    swanlab.init(project="multimodal-ner", run_name=config.run_name)

    device = torch.device(config.device)
    tokenizer = RobertaTokenizer.from_pretrained(config.text_encoder)
    processor = CLIPProcessor.from_pretrained(config.image_encoder)

    train_dataset = MultimodalNERDataset(config.train_file, tokenizer, processor, max_length=config.max_len)
    val_dataset = MultimodalNERDataset(config.val_file, tokenizer, processor, max_length=config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultimodalNER().to(device)

    # 分模块学习率
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" in n],
            "lr": config.fin_tuning_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "text_encoder" not in n],
            "lr": config.downs_en_lr,
        }
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=config.downs_en_lr, weight_decay=config.weight_decay_rate)

    t_total = len(train_loader) // config.gradient_accumulation_steps * config.epochs
    warmup_steps = int(t_total * config.warmup_prop)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    best_f1 = 0.0
    patience_counter = 0

    save_dir = f"save_models/{datetime.now().strftime('%Y-%m-%d')}_train-{config.dataset_name}_lr{config.lr}_bs{config.batch_size}"

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

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(train_loader)
        swanlab.log({"train/loss": avg_loss, "epoch": epoch})
        print(f"\n✅ Epoch {epoch} Train Loss: {avg_loss:.4f}")

        f1, report = evaluate(model, val_loader, device)
        print(f"🎯 Eval F1: {f1:.4f}")

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

    config = get_config()
    train(config)

