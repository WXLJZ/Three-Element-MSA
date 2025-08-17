# -*- coding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5ForConditionalGeneration, T5Tokenizer

from instruct.inst_construct import train_val_data_process, test_data_process
from models.T5_model import T5FineTuner
from models.LinearModel import LinearModel
from utils.logger import get_logger
from utils.data_utils import MetaphorDataset
from utils.evaluation import evaluate_test_data, evaluate_by_all_checkpoints
from utils.train_utils import SaveSentimentFigure, SaveMetaphorTypeFigure, LoggingCallback

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)

def main(args):
    # instruction construction
    logger.info(f"Instruction construction starting...")
    train_val_data_process(args)
    test_data_process(args)

    # show one sample to check the code and the expected output
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # Get example from the train set
    dataset = MetaphorDataset(tokenizer=tokenizer, data_dir=args.data_dir, dataset_name=args.dataset,
                              data_type='train', max_len=args.max_seq_length, truncate=args.truncate)
    data_sample = dataset[0]

    # sanity check
    # show one sample to check the code and the expected output format are correct
    print("\n======================================== Data Check =====================================================")
    print(f"Here is an example (from the train set):")
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
    print('num_train_epochs:',args.num_train_epochs)
    print("============================================================================================================\n")

    # training process
    if args.do_train:
        print("\n========================================= Conduct training ===========================================")
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, device_map="auto")
        tfm_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model is loaded from {args.model_name_or_path}")
        logger.info(f"Training arguments: {args}")

        sentiment_model = LinearModel()
        metaphor_type_model = LinearModel()

        model = T5FineTuner(args, tfm_model, tokenizer, sentiment_model, metaphor_type_model)

        # set ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=args.temp_dir + '/{epoch:02d}',
            save_top_k=5,
            monitor='val_micro_f1',
            mode='max',
            verbose=True,
            save_last=False,
            save_weights_only=True
        )

        # prepare for trainer
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
            logger=False,
            checkpoint_callback=checkpoint_callback,
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Save the sentiment, and metaphor-type representations
        tsne_dict = model.tsne_dict
        SaveMetaphorTypeFigure(args, tsne_dict)
        SaveSentimentFigure(args, tsne_dict)

        logger.info("Finish training and saving the model!")
    if args.do_eval_all:
        print("\n=============================== Evaluation in Test dataset by all checkpoints================================")
        evaluate_by_all_checkpoints(args, data_type='test')
    if args.do_eval:
        print("\n============================ Conduct inference on the best trained checkpoint ==================================")

        # initialize the T5 model from previous checkpoint
        logger.info(f"Load trained model from {args.output_dir}")
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

        sentiment_model = LinearModel()
        metaphor_type_model = LinearModel()
        model = T5FineTuner(args, tfm_model, tokenizer, sentiment_model, metaphor_type_model)

        # load the test dataset
        test_dataset = MetaphorDataset(tokenizer=tokenizer, data_dir=args.data_dir, dataset_name=args.dataset,
                                       data_type='test',max_len=args.max_seq_length, truncate=args.truncate)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

        # compute the performance scores
        scores = evaluate_test_data(test_loader, model, tokenizer, dataset_name=args.dataset, result_file=args.result_file)
        result_file_path = os.path.join(args.output_dir, "result_metric.jsonl")
        with open(result_file_path, 'a', encoding='utf-8') as f:
            scores['epoch_setting'] = args.num_train_epochs
            scores['setting'] = vars(args)
            json_line = json.dumps(scores, ensure_ascii=False) + '\n'
            f.write(json_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training and evaluation of the T5 model for the ABSA")
    # basic settings
    parser.add_argument("--data_dir", default='../data', type=str,help="The directory of the dataset")
    parser.add_argument("--temp_dir", default='../temp-models', type=str, help="The directory to save the temporary files")
    parser.add_argument("--dataset", default='Laptop', type=str, help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--retrieve_model_path", type=str, help="The name of the model for retrieval")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run inference with trained checkpoints", default=True)
    parser.add_argument("--do_eval_all", action='store_true', help="Whether to run inference with all checkpoints")
    parser.add_argument("--ICL", action='store_true', help="Whether to run ICL construction", default=False)
    parser.add_argument("--CoT", action='store_true', help="Whether to run CoT construction", default=False)
    parser.add_argument("--MTL", action='store_true', help="Whether to run MT construction", default=False)
    parser.add_argument("--result_file", type=str, help="The save path of generate results")

    # other parameters
    parser.add_argument("--max_seq_length", default=384, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--learning_rate", default=8e-5, type=float)
    parser.add_argument("--num_train_epochs", default=15000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=8e-5, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument('--truncate', action='store_true', default=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.1)
    parser.add_argument("--cont_temp", type=float, default=0.1)

    args = parser.parse_args()

    # Mannually set the output directory
    output_dir = Path('./outputs') / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    args.temp_dir = os.path.join(args.temp_dir, args.dataset)
    os.makedirs(args.temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.result_file = os.path.join(args.output_dir, f"Epoch_{args.num_train_epochs}_Time_{timestamp}.jsonl")

    # manage thenumber of files in the temp_dir
    file_list = os.listdir(args.temp_dir)
    if len(file_list) > 5:
        for file in file_list:
            os.remove(os.path.join(args.temp_dir, file))
            logger.info(f"Remove the file: {file} for the limited number of saved files in the temp_dir.")

    main(args)
