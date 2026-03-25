CHECKPOINTS_DIR=""
START=0
STEP=20
END=200

# conda activate verl
for i in $(seq $START $STEP $END); do
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir $CHECKPOINTS_DIR/global_step_$i/actor \
        --target_dir $CHECKPOINTS_DIR/global_step_$i/actor
done
