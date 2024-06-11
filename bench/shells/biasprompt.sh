#!/usr/bin/env bash

idx=`expr $1 % 2`
cuda=`expr $1 % 4`
echo "idx=$idx running on cuda:$cuda"
devfiles=$(ls cached_sample_data/device/*1.json)
statefiles=$(ls cached_sample_data/state/*.json)
i=0
ignores=()
datasetname=openImg
algorithm=BiasPrompt
rounds=3000
seed=42
ddl=120
for sf in $statefiles
do
    for df in $devfiles
    do
        n=`expr $i % 2`
        df=$(basename $df)
        sf=$(basename $sf)
        
        for ig in ${ignores[@]}
        do
            if [[ "${df:18:6}_${sf:9:8}" == $ig ]]
            then
                echo "${df:18:6}_${sf:9:8}"
                echo $ig
                echo "pass"
                ((i++))
                continue 2
            fi
        done
        

        if [ $n == $idx ]
        then
            # echo cached_sample_data/cvpr_inteplay_sets/device/$df;
            # echo cached_sample_data/cvpr_inteplay_sets/state/$sf;
            echo "Device: ${df%.*} State: ${sf%.*}";
            echo "python main.py --config configs/default.cfg --seed $seed --save_name Test_${algorithm}_${datasetname}_d${df%.*}_s${sf%.*} --gpu cuda:$cuda --dataset_name ${datasetname} --data_path /home/liangqqu/edgecase/fedavgmodels/data/Retina --num_rounds ${rounds} --deadline ${ddl} --device_path cached_sample_data/device/$df --state_path cached_sample_data/state/$sf --sub_dataset_size 1000 --iid --biasprompt --prompt_length 12 "
            # python main.py --config configs/ablation/interplay_heter.cfg --seed $seed --save_name CVPR_Interplay_Heter${algorithm}_${datasetname}_d${df%.*}_s${sf%.*} --gpu cuda:$cuda --dataset_name ${datasetname} --rounds ${rounds} --deadline ${ddl} --mp_dev --fix_mean --device_path cached_sample_data/cvpr_inteplay_sets/device/$df --behav_sample cached_sample_data/cvpr_inteplay_sets/state/$sf --correct --non_iid_clients_num 100 --ignore_train_time --test_curve --sub_dataset_size 1000 --iid --biasprompt --pcgrad
            python main.py --config configs/default.cfg --seed $seed --save_name Test_${algorithm}_${datasetname}_d${df%.*}_s${sf%.*} --gpu cuda:$cuda --dataset_name ${datasetname} --data_path /home/liangqqu/edgecase/fedavgmodels/data/Retina --num_rounds ${rounds} --deadline ${ddl} --device_path cached_sample_data/device/$df --state_path cached_sample_data/state/$sf --sub_dataset_size 1000 --iid --biasprompt --prompt_length 12
        fi
        ((i++))
    done
done
