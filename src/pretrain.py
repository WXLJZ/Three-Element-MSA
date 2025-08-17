# coding:utf-8
import json
import os
import gc
import re
import os
import sys
import pickle
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
from argparse import ArgumentParser
from collections import defaultdict
from transformers import BertTokenizer, AdamW, BertTokenizerFast, BertConfig, BertForMaskedLM
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler

from utils.logger import get_logger

sys.path.append('src')
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_data(args, tokenizer: BertTokenizer) -> dict:
    with open(args.pretrain_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'loading data path: {args.pretrain_data_path}')

    inputs = defaultdict(list)

    for entry in tqdm(data):
        sentence = entry['input']
        source = entry['Source']
        target = entry['Target']

        combined_text = f"{sentence} [SEP] {source} [SEP] {target}"
        inputs_dict = tokenizer.encode_plus(combined_text, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

        # Manually set token_type_ids
        sep_index = inputs_dict['input_ids'].index(tokenizer.sep_token_id)
        inputs_dict['token_type_ids'] = [0] * (sep_index + 1) + [1] * (len(inputs_dict['input_ids']) - sep_index - 1)

        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

        # print(f"decoded Text: {tokenizer.decode(inputs_dict['input_ids'])}")
        # print(f"input_ids: {inputs_dict['input_ids']}")
        # print(f"token_type_ids: {inputs_dict['input_ids']}")
        # print(f"attention_mask: {inputs_dict['attention_mask']}")
    print(f"============================================== Data Checking =============================================")
    print(f"The input data example:\n decoded Text: {tokenizer.decode(inputs['input_ids'][0])} \n input_ids: {inputs['input_ids'][0]} \n token_type_ids: {inputs['token_type_ids'][0]} \n attention_mask: {inputs['attention_mask'][0]}")
    print(f"============================================== Data Checking =============================================")
    os.makedirs(os.path.dirname(args.data_cache_path), exist_ok=True)
    save_pickle(inputs, args.data_cache_path)

    return inputs


class MetaphorDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class MetaphorDataCollator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])

            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


def load_data(args, tokenizer):
    with open(args.data_cache_path, 'rb') as f:
        train_data = pickle.load(f)

    collate_fn = MetaphorDataCollator(args.max_seq_len, tokenizer)
    train_dataset = MetaphorDataset(train_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)
    return train_dataloader


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_model_path, do_lower_case=True)
    model_config = BertConfig.from_pretrained(args.pretrain_model_path)
    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_path,
                                            config=model_config)

    model.to(args.device)

    return tokenizer, model


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def create_dirs(path):
    os.makedirs(path, exist_ok=True)


def save_model(args, model, tokenizer, global_steps, is_last=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_to_save = model.module if hasattr(model, 'module') else model
    # Ensure all tensors are contiguous
    for param in model_to_save.parameters():
        param.data = param.data.contiguous()

    if is_last:
        model_save_path = os.path.join(args.save_path, f'checkpoint-{global_steps}')
    else:
        model_save_path = os.path.join(args.record_save_path, f'checkpoint-{global_steps}')
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_vocabulary(model_save_path)

    logger.info(f'model saved in : {model_save_path} .')


def sorted_checkpoints(args, best_model_checkpoint, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(args.record_save_path).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    # Make sure we don't delete the best model.
    if best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
            checkpoints_sorted[-1],
            checkpoints_sorted[best_model_index],
        )
    return checkpoints_sorted


def rotate_checkpoints(args, best_model_checkpoint, use_mtime=False) -> None:
    if args.save_total_limit is None or args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, best_model_checkpoint, use_mtime=use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        shutil.rmtree(checkpoint)


def pretrain(args):
    logger.info('start pretraining ... ...')
    logger.info(f'loading from pretrain model path: {args.pretrain_model_path}')

    tokenizer, model = build_model_and_tokenizer(args)

    # if not os.path.exists(os.path.join(args.data_cache_path)):
    read_data(args, tokenizer)

    train_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    if args.fp16:
        scaler = GradScaler()

    pretrain_loss_list, global_steps_list = [], []

    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for step, batch in enumerate(train_iterator):
            batch_cuda = batch2cuda(args, batch)

            if args.fp16:
                with autocast():
                    loss, logits = model(**batch_cuda)[:2]
                scaler.scale(loss).backward()
            else:
                loss, logits = model(**batch_cuda)[:2]
                loss.backward()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            pretrain_loss_list.append(loss.item())
            global_steps_list.append(global_steps + 1)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                if (global_steps + 1) % args.logging_step == 0:
                    epoch_avg_loss = cur_avg_loss / args.logging_step
                    global_avg_loss = total_loss / (global_steps + 1)

                    logger.info(f"epoch - {epoch},  global steps - {global_steps + 1}, "
                          f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                    cur_avg_loss = 0.0
                global_steps += 1

                lr = scheduler.get_last_lr()[0]
                train_iterator.set_postfix_str(f'loss : {loss.item():.4f}, lr : {lr}, global steps : {global_steps} .')

            if (global_steps + 1) % args.save_steps == 0:
                save_model(args, model, tokenizer, global_steps)
                last_checkpoint_save_path = os.path.join(args.record_save_path, f'checkpoint-{global_steps}')
                rotate_checkpoints(args, last_checkpoint_save_path, use_mtime=False)

    logger.info('saving model at last epoch ... ...')
    save_model(args, model, tokenizer, global_steps, True)

    fig, ax = plt.subplots()
    ax.plot(global_steps_list, pretrain_loss_list, 'k-', label='pretrain_loss')
    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')

    fig_save_path = os.path.join(args.save_path, 'train_loss_curve.jpg')
    plt.savefig(fig_save_path)
    plt.show()

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = ArgumentParser(description='Pretrain BERT model on metaphor data')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--pretrain_data_path', type=str, default='../data/CMSA/train.json')
    parser.add_argument('--pretrain_model_path', type=str, default=f'/xxx/bert-base-chinese')
    parser.add_argument('--vocab_path', type=str, default=f'/xxx/bert-base-chinese/vocab.txt')
    parser.add_argument('--data_cache_path', type=str, default=f'../checkpoints/data/CMSA/pretrain.pkl')
    parser.add_argument('--save_path', type=str, default='../checkpoints/models/CMSA/pretrained_model')
    parser.add_argument('--record_save_path', type=str, default='../checkpoints/models/CMSA/pretrained_model_record')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_steps', type=int, default=20000)
    parser.add_argument('--save_total_limit', type=int, default=10)
    parser.add_argument('--logging_step', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--fp16', type=str, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_name', type=str, default='CMSA')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    args.n_gpus = torch.cuda.device_count()
    args.batch_size *= args.n_gpus

    create_dirs(args.save_path)
    create_dirs(args.record_save_path)

    seed_everything(args.seed)

    pretrain(args)


if __name__ == '__main__':
    main()