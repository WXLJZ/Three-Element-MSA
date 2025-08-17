import torch
from tqdm import tqdm
import re
import os
import json
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.data_utils import MetaphorDataset
from utils.logger import get_logger
from models.T5_model import T5FineTuner
from models.LinearModel import LinearModel

logger = get_logger(__name__)

def parser_outputs(outputs):
    """
    Parse the outputs
    """
    pattern = r'\[(.*?)\]'
    results = []
    for s in outputs:
        match = re.findall(pattern, s)
        if match:
            result = match[0].strip()
            results.append(result)
        else:
            result = ""
            results.append(result)
    return results

def evaluate_test_data(data_loader, model, tokenizer, dataset_name, result_file):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()
    preds, trues = [], []
    index= 0
    with open(result_file, 'a', encoding='utf-8') as f:
        for batch in tqdm(data_loader, desc="Evaluating"):
            pred_output = model.model.generate(input_ids=batch['source_ids'].to(device),
                                        attention_mask=batch['source_mask'].to(device),
                                        max_length=128)

            pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_output]
            true = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
            input = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]

            for input_text, p, t in zip(input, pred, true):
                index += 1
                record = {
                    'id': index,
                    "input": input_text,
                    "prediction": p,
                    "true": t
                }
                # print(f"Original Prediction: {p}, Original True: {t}")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            pred = parser_outputs(pred)
            true = parser_outputs(true)
            # for p, t in zip(pred, true):
            #     print(f"Prediction: {p}, True: {t}")

            preds.extend(pred)
            trues.extend(true)

    scores = compute_scores(preds, trues, dataset_name)
    logger.info(f"Scores: {scores}")
    return scores

def evaluate_by_all_checkpoints(args, data_type):
    all_scores = {}
    file_names = os.listdir(args.temp_dir)
    for i, file_name in enumerate(file_names):
        epoch = int(re.search(r'epoch=(\d+)', file_name).group(1))
        checkpoint_path = os.path.join(args.temp_dir, file_name)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Epoch {epoch}, model is loaded from {checkpoint_path}")
        sentiment_model = LinearModel()
        metaphor_type_model = LinearModel()

        t5_model = T5FineTuner(args, model, tokenizer, sentiment_model, metaphor_type_model)
        checkpoint = torch.load(checkpoint_path)
        t5_model.load_state_dict(checkpoint['state_dict'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_model.to(device)

        dataset = MetaphorDataset(tokenizer=tokenizer, data_dir=args.data_dir, dataset_name=args.dataset,
                                      data_type=data_type, max_len=args.max_seq_length, truncate=args.truncate)
        loader = DataLoader(dataset, batch_size=32, num_workers=4)

        all_scores[epoch] = evaluate_test_data(loader, t5_model, tokenizer, dataset_name=args.dataset, result_file=args.result_file)
    for epoch, scores in all_scores.items():
        logger.info(f"Epoch {epoch} in {data_type} dataset scores: {scores}")
    return all_scores

def compute_scores(predictions, trues, dataset_name):
    """
    Compute scores given the predictions and gold labels
    """
    assert len(predictions) == len(trues)
    if dataset_name == 'CMSA':
        pos_true, pos_pred, pos_right = 0, 0, 0
        neu_true, neu_pred, neu_right = 0, 0, 0
        neg_true, neg_pred, neg_right = 0, 0, 0
        for pred, true in zip(predictions, trues):
            pred_sentiment = pred
            true_sentiment = true
            if true_sentiment == '正向':
                pos_true += 1
            elif true_sentiment == '中性':
                neu_true += 1
            elif true_sentiment == '负向':
                neg_true += 1
            # calculate the sentiment [prediction]
            if pred_sentiment == '正向':
                pos_pred += 1
            elif pred_sentiment == '中性':
                neu_pred += 1
            elif pred_sentiment == '负向':
                neg_pred += 1
            # calculate the sentiment [right]
            if true_sentiment == pred_sentiment:
                if true_sentiment == '正向':
                    pos_right += 1
                elif true_sentiment == '中性':
                    neu_right += 1
                elif true_sentiment == '负向':
                    neg_right += 1
    elif dataset_name == 'EMSA':
        pos_true, pos_pred, pos_right = 0, 0, 0
        neu_true, neu_pred, neu_right = 0, 0, 0
        neg_true, neg_pred, neg_right = 0, 0, 0
        for pred, true in zip(predictions, trues):
            pred_sentiment = pred
            true_sentiment = true
            # calculate the sentiment [truth]
            if true_sentiment == 'positive':
                pos_true += 1
            elif true_sentiment == 'neutral':
                neu_true += 1
            elif true_sentiment == 'negative':
                neg_true += 1
            # calculate the sentiment [prediction]
            if pred_sentiment == 'positive':
                pos_pred += 1
            elif pred_sentiment == 'neutral':
                neu_pred += 1
            elif pred_sentiment == 'negative':
                neg_pred += 1
            # calculate the sentiment [right]
            if true_sentiment == pred_sentiment:
                if true_sentiment == 'positive':
                    pos_right += 1
                elif true_sentiment == 'neutral':
                    neu_right += 1
                elif true_sentiment == 'negative':
                    neg_right += 1

    accuracy = (pos_right + neu_right + neg_right) / (pos_true + neu_true + neg_true) if pos_true + neu_true + neg_true != 0 else 0
    pos_p = pos_right / pos_pred if pos_pred != 0 else 0
    pos_r = pos_right / pos_true if pos_true != 0 else 0
    pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r) if pos_p + pos_r != 0 else 0
    neu_p = neu_right / neu_pred if neu_pred != 0 else 0
    neu_r = neu_right / neu_true if neu_true != 0 else 0
    neu_f1 = 2 * neu_p * neu_r / (neu_p + neu_r) if neu_p + neu_r != 0 else 0
    neg_p = neg_right / neg_pred if neg_pred != 0 else 0
    neg_r = neg_right / neg_true if neg_true != 0 else 0
    neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r) if neg_p + neg_r != 0 else 0
    micro_f1 = (pos_f1 + neu_f1 + neg_f1) / 3
    # 输出结果，输出百分数，保留两位小数
    # print(f"\t\t ACCURACY: {accuracy:.2%}")
    # print(f"\t\t positive: precision={pos_p:.2%}, recall={pos_r:.2%}, f1={pos_f1:.2%}")
    # print(f"\t\t neutral: precision={neu_p:.2%}, recall={neu_r:.2%}, f1={neu_f1:.2%}")
    # print(f"\t\t negative: precision={neg_p:.2%}, recall={neg_r:.2%}, f1={neg_f1:.2%}")
    # print(f"\t\t Micro F1: {micro_f1:.2%}")
    return {"accuracy": accuracy,
            "positive": f"precision={pos_p:.2%}, recall={pos_r:.2%}, f1={pos_f1:.2%}",
            "neutral": f"precision={neu_p:.2%}, recall={neu_r:.2%}, f1={neu_f1:.2%}",
            "negative": f"precision={neg_p:.2%}, recall={neg_r:.2%}, f1={neg_f1:.2%}",
            "micro_f1": micro_f1}


