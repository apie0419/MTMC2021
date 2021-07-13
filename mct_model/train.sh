#!/bin/bash

filename="baseline+RW+cam+ranked2.5m1.5T0_r0.5c1_nofc"
weight="/home/apie/projects/MTMC2021_ver2/mct_model/checkpoints/${filename}_4.pth"
expfilename="baseline+RW+cam+ranked2.5m1.5T0_r0.5c1_nofc_th_exp"

if [[ ! -e /home/apie/projects/MTMC2021_ver2/mct_model/logs ]]; then
    mkdir /home/apie/projects/MTMC2021_ver2/mct_model/logs
fi

# rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename.txt
# rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt
rm -fr /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt

# python train.py -o "$filename" | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$filename\_train.txt

for i in `seq 0.65 0.05 0.9`;
do  
    cd /home/apie/projects/MTMC2021_ver2/pipeline
    python step4_mutli_camera_tracking.py -w "$weight" -s $i > /dev/null 2>&1
    python step5_merge_results.py
    cd /home/apie/projects/MTMC2021_ver2/eval
    mv /home/apie/projects/MTMC2021_ver2/dataset/validation/track3.txt .
    echo Theshold "$i": | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt
    python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/apie/projects/MTMC2021_ver2/mct_model/logs/$expfilename.txt
done