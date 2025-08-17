import faiss
import json
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)

def components_tokenizer(data, tokenizer: AutoTokenizer, max_seq_len):
    inputs = defaultdict(list)
    for entry in data:
        sentence = entry['input']
        source = entry['Source']
        target = entry['Target']
        combined_text = f"{sentence} [SEP] {source} [SEP] {target}"

        inputs_dict = tokenizer.encode_plus(combined_text, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

        # Manually set token_type_ids
        sep_index = inputs_dict['input_ids'].index(tokenizer.sep_token_id)
        inputs_dict['token_type_ids'] = [0] * (sep_index + 1) + [1] * (len(inputs_dict['input_ids']) - sep_index - 1)

        # Pad and truncate
        input_ids = torch.zeros(max_seq_len, dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        seq_len = len(inputs_dict['input_ids'])
        if seq_len <= max_seq_len:
            input_ids[:seq_len] = torch.tensor(inputs_dict['input_ids'], dtype=torch.long)
            token_type_ids[:seq_len] = torch.tensor(inputs_dict['token_type_ids'], dtype=torch.long)
            attention_mask[:seq_len] = torch.tensor(inputs_dict['attention_mask'], dtype=torch.long)
        else:
            input_ids[:max_seq_len] = torch.tensor(inputs_dict['input_ids'][:max_seq_len - 1] + [tokenizer.sep_token_id], dtype=torch.long)
            token_type_ids[:max_seq_len] = torch.tensor(inputs_dict['token_type_ids'][:max_seq_len], dtype=torch.long)
            attention_mask[:max_seq_len] = torch.tensor(inputs_dict['attention_mask'][:max_seq_len], dtype=torch.long)

        inputs['input_ids'].append(input_ids)
        inputs['token_type_ids'].append(token_type_ids)
        inputs['attention_mask'].append(attention_mask)

    inputs['input_ids'] = torch.stack(inputs['input_ids'])
    inputs['token_type_ids'] = torch.stack(inputs['token_type_ids'])
    inputs['attention_mask'] = torch.stack(inputs['attention_mask'])
    return inputs

class Retriever():
    def __init__(self, retrieve_path, retrieve_model_path, retrieve_nums):
        self.retrieve_path = retrieve_path
        self.retrieve_nums = retrieve_nums

        with open(retrieve_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.sents = []
        self.labels = []
        self.sources = []
        self.targets = []

        for d in self.data:
            self.sents.append(d['input'])
            self.labels.append(d['output'])
            self.sources.append(d['Source'])
            self.targets.append(d['Target'])

        self.data_dict = {}
        for sent, label, source, target in zip(self.sents, self.labels, self.sources, self.targets):
            self.data_dict[sent] = (label, source, target)

        # Initialize models
        logger.info("Initializing retrieval model...")
        logger.info("Loading retrieval model from {}".format(retrieve_model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(retrieve_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(retrieve_model_path)
        self.init_embeddings(self.data)

    def init_embeddings(self, data):
        logger.info("Initializing embeddings...")
        self.embeddings = self.encode(data)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def encode(self, data, batch_size=512):
        all_embeddings = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            encoded_input = components_tokenizer(batch_data, self.tokenizer, 512)
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def retrieve(self, query):
        query_embedding = self.encode([query])
        distances, indices = self.index.search(query_embedding, self.index.ntotal)
        sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=False)
        # Getting the top k results, excluding the input example
        top_results = sorted_results[1: 1 + self.retrieve_nums]
        # Sort the selected examples from greatest to smallest distance, so that the most similar examples are closest to the input.
        top_results.reverse()
        res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in top_results]
        return res
