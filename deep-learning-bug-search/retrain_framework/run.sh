#!bin/sh

TARGET=OBJ_obj2txt
retrain_id=1
epoch_num=10
copy_num=10
python gen_add_data.py --target=$TARGET --src=./selected.list --id=$retrain_id --copies=$copy_num
cd retrain && python train.py --epoch=$epoch_num --save_path=./saved_model/graphnn-model-iter$retrain_id --add_data_time=$retrain_id && cd ..
#python test_pair.py --target=$TARGET --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --src=./output/selected.txt   ## for test
python vul_embed.py --target=$TARGET --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --retrain_id=$retrain_id
python get_score.py --target=$TARGET --thread=8 --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --retrain_id=$retrain_id
