LOCAL_DIR="./raw_data"
mkdir $LOCAL_DIR
#download deepscaler
bash hfd.sh agentica-org/DeepScaleR-Preview-Dataset --local-dir $LOCAL_DIR/deepscaler --dataset

#download livecodebench
bash hfd.sh livecodebench/code_generation_lite --local-dir $LOCAL_DIR/livecodebench --dataset

#download webInstruct
bash hfd.sh TIGER-Lab/WebInstruct-verified --local-dir $LOCAL_DIR/webinstruct --dataset

#download medbullets
bash hfd.sh mkieffer/Medbullets --local-dir $LOCAL_DIR/medbullets --dataset

#download MedXpertQA
bash hfd.sh TsinghuaC3I/MedXpertQA --local-dir $LOCAL_DIR/MedXpertQA --dataset

#download CommonsenseQA
bash hfd.sh tau/commonsense_qa --local-dir $LOCAL_DIR/commonsense_qa --dataset
