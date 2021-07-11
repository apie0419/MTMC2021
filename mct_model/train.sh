#!/bin/bash

filename="baseline+RW+ranked1.4m1.2T0_all"
weight="/home/apie/projects/MTMC2021_ver2/mct_model/checkpoints/${filename}_5.pth"
expfilename="baseline+RW+ranked1.4m1.2T0_all_th_exp"

if [[ ! -e /home/apie/projects/MTMC2021_ver2/mct_model/logs ]]; then
    mkdir /home/apie/projects/MTMC2021_ver2/mct_model/logs
fi

# rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
# rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt
rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt
# 
# python train.py -o "$filename" | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt

for i in `seq 0.6 0.05 0.8`;
do  
    cd /home/apie/projects/MTMC2021_ver2/pipeline
    python step4_mutli_camera_tracking.py -w "$weight" -s $i > /dev/null 2>&1
    python step5_merge_results.py
    cd /home/apie/projects/MTMC2021_ver2/eval
    mv /home/apie/projects/MTMC2021_ver2/dataset/validation/track3.txt .
    echo Theshold "$i": | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt
    python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt
done