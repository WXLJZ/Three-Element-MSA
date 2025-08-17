import json
import torch
import os
from tqdm import tqdm

from utils.logger import get_logger
from instruct.retriever import Retriever
from instruct.inst_template import emsa_icl_template, emsa_template, emsa_cot_template, emsa_mtl_template, cmsa_icl_template, cmsa_template, cmsa_cot_template

logger = get_logger(__name__)

def construct_inst(json_path, save_path, retriever=None, args=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Instruction Construction"):
        if args.ICL:
            demonstrations = retriever.retrieve(entry)
            demonstrations_str = ""
            for demo in demonstrations:
                if args.dataset == 'EMSA':
                    demonstrations_str += f"Input: [{demo[0]}] | Source component: [{demo[1][1]}] | Target component: [{demo[1][2]}]\n" \
                                          f"Output: The sentiment is [{demo[1][0]}]\n"
                elif args.dataset == 'CMSA':
                    demonstrations_str += f"输入：[{demo[0]}] | 源成分：[{demo[1][1]} | 目标成分：[{demo[1][2]}]\n" \
                                          f"输出：情感极性是[{demo[1][0]}]\n"
                else:
                    raise ValueError(f"Invalid dataset name: {args.dataset}")

            if args.CoT:
                template = emsa_cot_template if args.dataset == 'EMSA' else cmsa_cot_template
                inst = template.format(
                    input=entry['input'],
                    target=entry['Target'],
                    source=entry['Source'],
                    demonstration=demonstrations_str
                )
            else:
                template = emsa_icl_template if args.dataset == 'EMSA' else cmsa_icl_template
                inst = template.format(
                    input=entry['input'],
                    target=entry['Target'],
                    source=entry['Source'],
                    demonstration=demonstrations_str
                )

            entry['instruction'] = inst
            entry['demonstration'] = demonstrations_str
        else:
            template = emsa_template if args.dataset == 'EMSA' else cmsa_template
            inst = template.format(
                input=entry['input'],
                target=entry['Target'],
                source=entry['Source']
            )
            entry['instruction'] = inst


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def train_val_data_process(args):
    logger.info("Dataset {} , constructing train & validation instruction ...".format(args.dataset))
    inst_data_dir = os.path.join(args.data_dir, args.dataset, 'inst_data')
    os.makedirs(inst_data_dir, exist_ok=True)
    # demonstrations file path
    demo_file = os.path.join(args.data_dir, args.dataset, 'train.json')
    # train data file path
    train_file = os.path.join(args.data_dir, args.dataset, 'train.json')
    # train instruction file path
    inst_train_file = os.path.join(inst_data_dir, 'inst_train.json')
    # val data file path
    val_file = os.path.join(args.data_dir, args.dataset, 'val.json')
    # val instruction file path
    inst_val_file = os.path.join(inst_data_dir, 'inst_val.json')

    if args.ICL:
        retriever = Retriever(retrieve_path=demo_file, retrieve_model_path=args.retrieve_model_path, retrieve_nums=1)
    else:
        retriever = None
    logger.info("Train dataset...")
    construct_inst(train_file, inst_train_file, retriever=retriever, args=args)
    logger.info("Validation dataset...")
    construct_inst(val_file, inst_val_file, retriever=retriever, args=args)
    logger.info("Dataset {} , constructing train & validation instruction done.".format(args.dataset))
    del retriever
    torch.cuda.empty_cache()

def test_data_process(args):
    logger.info("Dataset {} , constructing test instruction ...".format(args.dataset))
    inst_data_dir = os.path.join(args.data_dir, args.dataset, 'inst_data')
    os.makedirs(inst_data_dir, exist_ok=True)
    # demonstrations file path
    demo_file = os.path.join(args.data_dir, args.dataset, 'train.json')
    # test data file path
    test_file = os.path.join(args.data_dir, args.dataset, 'test.json')
    # test instruction file path
    inst_test_file = os.path.join(inst_data_dir, 'inst_test.json')

    if args.ICL:
        retriever = Retriever(retrieve_path=demo_file, retrieve_model_path=args.retrieve_model_path, retrieve_nums=1)
    else:
        retriever = None
    construct_inst(test_file, inst_test_file, retriever=retriever, args=args)
    logger.info("Dataset {} , constructing test instruction done.".format(args.dataset))
    del retriever
    torch.cuda.empty_cache()

    with open(inst_test_file, 'r', encoding='utf-8') as f:
        inst_test_data = json.load(f)

    return inst_test_data