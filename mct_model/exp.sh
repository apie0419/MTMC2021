#!/bin/bash

weight="/home/apie/projects/MTMC2021_ver2/mct_model/checkpoints/baseline_3.pth"
filename="cosine+RW_th_exp"

for i in `seq 0.6 0.05 0.95`;
do  
    cd /home/apie/projects/MTMC2021_ver2/pipeline
    python step4_mutli_camera_tracking.py -w "$weight" -s $i > /dev/null 2>&1
    python step5_merge_results.py
    cd /home/apie/projects/MTMC2021_ver2/eval
    mv /home/apie/projects/MTMC2021_ver2/dataset/validation/track3.txt .
    echo Theshold "$i": | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
    python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
done