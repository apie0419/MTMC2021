#!/bin/bash

filename="baseline+RW"

if [[ ! -e /home/apie/projects/MTMC2021_ver2/mct_model/logs ]]; then
    mkdir /home/apie/projects/MTMC2021_ver2/mct_model/logs
fi

rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt

python train.py -o "$filename" | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt

for i in {1..5}
do    
    weight="/home/apie/projects/MTMC2021_ver2/mct_model/checkpoints/${filename}_${i}.pth"
    cd /home/apie/projects/MTMC2021_ver2/pipeline
    python step4_mutli_camera_tracking.py -w "$weight"
    python step5_merge_results.py
    cd /home/apie/projects/MTMC2021_ver2/eval
    mv /home/apie/projects/MTMC2021_ver2/dataset/validation/track3.txt .
    echo Epoch "$i": | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
    python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
done