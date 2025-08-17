from torch.utils.data import Dataset
import torch
import json

from utils.logger import get_logger

logger = get_logger(__name__)

class MetaphorDataset(Dataset):
    def __init__(self, tokenizer, data_dir, dataset_name, data_type, max_len=512, truncate=False):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.data_path = f'{data_dir}/{dataset_name}/inst_data/inst_{data_type}.json'
        self.max_len = max_len
        self.truncated_data_num = 0
        self.truncate = truncate
        self.inputs = []
        self.outputs = []
        self.contrastive_labels = {'sentiment': [], 'metaphor_type': []}
        self.load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.outputs[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.outputs[index]["attention_mask"].squeeze()  # might need to squeeze

        if self.data_type == 'train' or self.data_type == 'val':
            sentiment_label = torch.tensor(self.contrastive_labels['sentiment'][index])
            metaphor_type_label = torch.tensor(self.contrastive_labels['metaphor_type'][index])
            return {"source_ids": source_ids,
                    "source_mask": source_mask,
                    "target_ids": target_ids,
                    "target_mask": target_mask,
                    'sentiment_labels': sentiment_label,
                    'metaphor_type_labels': metaphor_type_label
                    }
        elif self.data_type == 'test':
            return {"source_ids": source_ids,
                    "source_mask": source_mask,
                    "target_ids": target_ids,
                    "target_mask": target_mask
                    }
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

    def load_data(self):
        self._build_examples()

    def _build_examples(self):
        # The number of truncated data in the dataset
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data:
            input = d['instruction']
            output = f"The sentiment is [{d['output']}]" if self.dataset_name == 'EMSA' else f"情感极性是[{d['output']}]"
            if len(input.split()) > self.max_len:
                self.truncated_data_num += 1
            if self.data_type == 'train' or self.data_type == 'val':
                sentiment_label = self.get_sentiment_label(d['output'])
                metaphor_type_label = self.get_metaphor_type_label(d['Type'])
                tokenized_input = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.max_len, padding="max_length",
                    truncation=self.truncate, return_tensors="pt"
                )
                # print(f"input: {input}")
                # print(f"tokenized_input: {tokenized_input}")
                # print(f"decoded_input: {self.tokenizer.decode(tokenized_input['input_ids'].squeeze())}")
                tokenized_output = self.tokenizer.batch_encode_plus(
                    [output], max_length=self.max_len, padding="max_length",
                    truncation=self.truncate, return_tensors="pt"
                )
                self.inputs.append(tokenized_input)
                self.outputs.append(tokenized_output)
                self.contrastive_labels['sentiment'].append(sentiment_label)
                self.contrastive_labels['metaphor_type'].append(metaphor_type_label)
            else:
                tokenized_input = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.max_len+256, padding="max_length",
                    truncation=self.truncate, return_tensors="pt"
                )
                tokenized_output = self.tokenizer.batch_encode_plus(
                    [output], max_length=self.max_len+256, padding="max_length",
                    truncation=self.truncate, return_tensors="pt"
                )
                self.inputs.append(tokenized_input)
                self.outputs.append(tokenized_output)

    def get_sentiment_label(self, label):
        if label == 'positive' or label == '正向':
            return 0
        elif label == 'negative' or label == '负向':
            return 1
        elif label == 'neutral' or label == '中性':
            return 2
        else:
            raise ValueError(f"Invalid sentiment label: {label}")

    def get_metaphor_type_label(self, label):
        if 'VERB' in label: # Verb metaphors
            return 0
        elif 'MODIFIERS' in label or 'ATTRIBUTE' in label: # attribute metaphors
            return 1
        elif 'NOUN' in label: # Noun metaphors
            return 2
        else:
            raise ValueError(f"Invalid metaphor type label: {label}")



