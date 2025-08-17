from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
import torch
import re
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from models.losses import SupConLoss
from utils.data_utils import MetaphorDataset
from utils.logger import get_logger

logger = get_logger(__name__)

def get_dataset(tokenizer, type_path, args):
    dataset = MetaphorDataset(tokenizer=tokenizer, data_dir=args.data_dir, dataset_name=args.dataset,
                               data_type=type_path, max_len=args.max_seq_length, truncate=args.truncate)
    logger.info(f"Loaded {type_path} dataset with {len(dataset)} examples, {dataset.truncated_data_num} examples are truncated.")
    return dataset

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer, sentiment_model, metaphor_type_model):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.sentiment_model = sentiment_model
        self.metaphor_type_model = metaphor_type_model

        self.tsne_dict = {
             'sentiment_vecs': [],
             'metaphor_type_vecs': [],
             'sentiment_labels': [],
             'metaphor_type_labels': []
             }

        self.valid_preds = []
        self.valid_trues = []

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )

        last_state = main_pred.encoder_last_hidden_state

        # sentiment contrastive loss
        sentiment_pred = self.sentiment_model(last_state, attention_mask)
        # metaphor type contrastive loss
        metaphor_type_pred = self.metaphor_type_model(last_state, attention_mask)
        # get final encoder layer representation
        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        return main_pred, sentiment_pred, metaphor_type_pred, pooled_encoder_layer

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, sentiment_pred, metaphor_type_pred, pooled_encoder_layer = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        sentiment_labels = batch['sentiment_labels']
        metaphor_type_labels = batch['metaphor_type_labels']

        # Calculate the characteristic-specific losses
        sentiment_summed = sentiment_pred
        sentiment_normed = normalize(sentiment_summed, p=2.0, dim=2)
        sentiment_contrastive_loss = criterion(sentiment_normed, sentiment_labels)
        # print('sentiment_loss:\t', sentiment_contrastive_loss)

        metaphor_type_summed = metaphor_type_pred
        metaphor_type_normed = normalize(metaphor_type_summed, p=2.0, dim=2)
        metaphor_type_contrastive_loss = criterion(metaphor_type_normed, metaphor_type_labels)
        # print('metaphor_type_loss:\t', metaphor_type_contrastive_loss)

        # Use these for the version without SCL (no characteristic-specific representations)
        sentiment_encs = sentiment_normed.detach().cpu().numpy()[:, 0].tolist()
        metaphor_type_encs = metaphor_type_normed.detach().cpu().numpy()[:, 0].tolist()

        sentiment_labs = sentiment_labels.detach().cpu().tolist()
        metaphor_type_labs = metaphor_type_labels.detach().cpu().tolist()

        self.tsne_dict['sentiment_vecs'] += sentiment_encs
        self.tsne_dict['metaphor_type_vecs'] += metaphor_type_encs

        self.tsne_dict['sentiment_labels'] += sentiment_labs
        self.tsne_dict['metaphor_type_labels'] += metaphor_type_labs

        # return original loss plus the characteristic-specific SCL losses
        # print('outputs[0]:\t', outputs[0])
        # print('outputs[0]*10:\t', outputs[0]*10)
        # print('sentiment_contrastive_loss + metaphor_type_contrastive_loss:\t', sentiment_contrastive_loss + metaphor_type_contrastive_loss)
        loss = outputs[0] * 20 + sentiment_contrastive_loss * 5 + metaphor_type_contrastive_loss * 2
        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        from utils.evaluation import parser_outputs
        loss = self._step(batch)

        # 生成预测文本和真实文本
        input_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        target_ids[target_ids[:, :] == -100] = self.tokenizer.pad_token_id

        # 生成时设置 eos_token_id 和 pad_token_id
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=batch["source_mask"],
            max_length=128
        )

        preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        trues = [self.tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]
        preds = parser_outputs(preds)
        trues = parser_outputs(trues)

        self.valid_preds.extend(preds)
        self.valid_trues.extend(trues)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        from utils.evaluation import compute_scores
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        if len(self.valid_preds) > 0:
            metrics = compute_scores(self.valid_preds, self.valid_trues, self.hparams.dataset)
        else:
            metrics = 0

        # To monitor the metrics, transfer the metrics to Tensor format
        val_accuracy = torch.tensor(metrics["accuracy"])
        val_micro_f1 = torch.tensor(metrics["micro_f1"])

        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": val_accuracy, "val_micro_f1": val_micro_f1}
        return {"avg_val_loss": avg_loss, "val_accuracy": metrics["accuracy"], "val_micro_f1": metrics["micro_f1"],
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs, "positive_metric": metrics["positive"],
                "negative_metric": metrics["negative"], "neutral_metric": metrics["neutral"]}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        sentiment_model = self.sentiment_model
        metaphor_type_model = self.metaphor_type_model

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }, {
                "params": [p for n, p in sentiment_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in sentiment_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in metaphor_type_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in metaphor_type_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)