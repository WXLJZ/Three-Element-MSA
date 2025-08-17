python main.py \
    --ICL \
    --dataset EMSA \
    --retrieve_model_path /xxx/checkpoints/models/EMSA/pretrained_model/checkpoint-15680 \
    --num_train_epochs 35 \
    --model_name_or_path /xxx/models/flan-t5-large \
    --do_eval_all \
    --do_train

python main.py \
    --ICL \
    --dataset CMSA \
    --retrieve_model_path /xxx/checkpoints/models/CMSA/pretrained_model/checkpoint-4700 \
    --num_train_epochs 15 \
    --model_name_or_path /xxx/models/T5-large-CSA \
    --do_eval_all \
    --do_train
