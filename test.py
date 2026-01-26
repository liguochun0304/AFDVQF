import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import evaluate_each_class, evaluate, get_chunks
from dataloader import MMPNERDataset, MMPNERProcessor, collate_fn, spans_to_bio, bio_to_spans
from model import build_model, EntityTarget

script_dir = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = "/root/autodl-fs"
DATA_ROOT = os.path.join(STORAGE_ROOT, "data")


def evaluate_model(model, val_loader, device, tags, type_names=None, label_mapping=None):
    model.eval()
    all_preds, all_labels, all_words = [], [], []
    idx2tag = {v: k for k, v in tags.items()}
    to_list = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    
    sample_details = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images)

            for pred_spans, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = int(mask.sum().item())
                seq_len = len(l_ids)
                l_ids_list = to_list(l_ids[:valid_len])
                
                entity_targets = []
                for span_tuple in pred_spans:
                    if isinstance(span_tuple, tuple) and len(span_tuple) >= 3:
                        s, e, t = span_tuple[0], span_tuple[1], span_tuple[2]
                        if (s >= 1 and e >= 1 and s < valid_len and e < valid_len and 
                            s <= e and t >= 0 and t < len(type_names)):
                            entity_targets.append(EntityTarget(start=s, end=e, type_id=t, region_id=-1))
                
                type_name2id = {name: i for i, name in enumerate(type_names)}
                gold_entities = bio_to_spans(l_ids_list, idx2tag, type_name2id)
                gold_spans = [(et.start, et.end, et.type_id) for et in gold_entities]
                
                p_ids = spans_to_bio(entity_targets, seq_len, type_names, label_mapping, ignore_special=True, valid_len=valid_len)
                p_ids = p_ids[:valid_len]

                kept_pred, kept_gold = [], []
                for pid, lid in zip(p_ids, l_ids_list):
                    tag_name = idx2tag.get(lid, "O")
                    if tag_name in ("[CLS]", "[SEP]", "X", "PAD"):
                        continue
                    kept_pred.append(pid)
                    kept_gold.append(lid)

                if len(kept_pred) != len(kept_gold):
                    min_len = min(len(kept_pred), len(kept_gold))
                    kept_pred = kept_pred[:min_len]
                    kept_gold = kept_gold[:min_len]

                all_preds.append(kept_pred)
                all_labels.append(kept_gold)
                all_words.append([])
                
                if len(sample_details) < 3:
                    pred_spans_list = [(et.start, et.end, et.type_id) for et in entity_targets]
                    pred_chunks = get_chunks(kept_pred, tags)
                    gold_chunks = get_chunks(kept_gold, tags)
                    sample_details.append({
                        'pred_spans': pred_spans_list,
                        'gold_spans': gold_spans,
                        'pred_bio': kept_pred,
                        'gold_bio': kept_gold,
                        'pred_chunks': pred_chunks,
                        'gold_chunks': gold_chunks
                    })
    
    print("=" * 60)
    print("测试2: 从spans转换到BIO序列评估")
    print("=" * 60)
    
    for i, detail in enumerate(sample_details):
        print(f"\n样本 {i+1}:")
        pred_spans_str = [(s, e, type_names[t]) for s, e, t in detail['pred_spans']]
        gold_spans_str = [(s, e, type_names[t]) for s, e, t in detail['gold_spans']]
        print(f"  预测spans: {pred_spans_str}")
        print(f"  真实spans: {gold_spans_str}")
        print(f"  预测BIO: {detail['pred_bio']}")
        print(f"  真实BIO: {detail['gold_bio']}")
        print(f"  预测chunks: {detail['pred_chunks']}")
        print(f"  真实chunks: {detail['gold_chunks']}")
    
    acc, f1, p, r = evaluate(all_preds, all_labels, all_words, tags)
    print(f"\n总体指标: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    
    if isinstance(next(iter(tags.keys())), int):
        tag_names = list(tags.values())
    else:
        tag_names = list(tags.keys())
    entity_types = sorted({name.split('-')[-1] for name in tag_names if '-' in name})
    for ent_type in entity_types:
        f1_c, p_c, r_c = evaluate_each_class(all_preds, all_labels, all_words, tags, ent_type)
        print(f"{ent_type}: P={p_c:.4f}, R={r_c:.4f}, F1={f1_c:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    return acc, f1, p, r


def load_config(model_dir):
    config_path = os.path.join(STORAGE_ROOT, "save_models", model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.save_name)
    config.device = args.device
    device = torch.device(config.device)

    DATA_PATH = {
        "twitter2015": {
            'train': os.path.join(DATA_ROOT, 'twitter2015/train.txt'),
            'valid': os.path.join(DATA_ROOT, 'twitter2015/valid.txt'),
            'test':  os.path.join(DATA_ROOT, 'twitter2015/test.txt'),
        },
        "twitter2017": {
            'train': os.path.join(DATA_ROOT, 'twitter2017/train.txt'),
            'valid': os.path.join(DATA_ROOT, 'twitter2017/valid.txt'),
            'test':  os.path.join(DATA_ROOT, 'twitter2017/test.txt'),
        }
    }
    IMG_PATH = {
        'twitter2015': os.path.join(DATA_ROOT, 'twitter2015/twitter2015_images'),
        'twitter2017': os.path.join(DATA_ROOT, 'twitter2017/twitter2017_images'),
    }

    img_path = IMG_PATH[config.dataset_name]
    data_path = DATA_PATH[config.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processor = MMPNERProcessor(data_path, config.text_encoder)

    test_dataset = MMPNERDataset(
        processor, transform,
        img_path=img_path, max_seq=config.max_len,
        sample_ratio=1.0, mode='test', set_prediction=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    tokenizer = processor.tokenizer
    type_names = test_dataset.type_names

    model = build_model(config, tokenizer=tokenizer, type_names=type_names).to(device)
    model_path = os.path.join(STORAGE_ROOT, "save_models", args.save_name, "model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    evaluate_model(
        model, test_loader, device, test_dataset.label_mapping,
        type_names=type_names,
        label_mapping=test_dataset.label_mapping
    )


if __name__ == "__main__":
    main()
