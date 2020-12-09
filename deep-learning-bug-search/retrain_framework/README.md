## Retrain Framework
The retraining framework works like this:

1. (Re)train the model with all the data in the training dataset.
2. Search in the firmware dataset to get a similarity score list.
3. **Manually** check the top of the list, and select those negative (or positive) examples to be added.
4. Add extra data into the training dataset according to the negative (or positive) examples.

The framework is to repeat the four steps. Since step 3 requires manual check, the bash file is written in order 4, 1, 2.
For each retraining iteration, one can first add the selected negative (or positive) examples in a text file, and run the bash script.

### Integreted bash file for execution:
`bash preprocess.sh`

This file only needs to be executed once.
It is for finding all the .acfgs files and record their locations in file `graphlist`.
This list will be used during the searching part, and all the files in this list will be considered in the file.
It also creates some directory for saving outputs.

`bash run.sh`

This is the script file of retraining, in the order step4, step1, step2.
In the file, `retrain_id` is the number of retraining times, and `TARGET` is the target vulnerability function. After each retraining epoch the retrain\_id should be increased by 1 manually.

### Detailed explanation of each python file
`python gen_add_data.py --target=$TARGET --src=./output/selected.txt --id=$retrain_id --copies=$copy_num`

+ This file is for adding training data.
It loads graph path from `src`.
For each graph g, it adds the pair (g, target) into the training dataset.
+ `--src` is the path of the list of manually selected files.
+ `--id` is the id of added files.
+ `--copies` is the number of copies of pair that are added into the training set.

`cd retrain && python train.py --epoch=$epoch_num --save_path=./saved_model/graphnn-model-iter$retrain_id --add_data_time=$retrain_id && cd ..`

+ This is the file for (re)training the model.
`--add_data_time` is the number of added data. For other detailed parameters, see `python train.py -h`.

`#python test_pair.py --target=$TARGET --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --src=./output/selected.txt   ## for test`

+ This is a file for testing.
It outputs the similarity score between target and the graphs in `src`. This can show whether the retrained model actually works on the added data.

`python vul_embed.py --target=$TARGET --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --retrain_id=$retrain_id`

+ This is the file for getting the embedded vector of the target vulnerability function. This process is separated from the searching because the target vector is used as a constant in the model of searching.
+ `--load_path` is the path for loading the model

`python get_score.py --target=$TARGET --thread=8 --load_path=./retrain/saved_model/graphnn-model-iter$retrain_id-$epoch_num --retrain_id=$retrain_id`

+ This is the file for searching over the firmware dataset.
+ The output will be in the directory `./output`
+ `--thread` is the number of thread used for searching.
+ For other detailed parameters, see `python get_score.py -h`
