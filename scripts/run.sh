tasks="AntMorphology-Exact-v0 DKittyMorphology-Exact-v0 Superconductor-RandomForest-v0 TFBind8-Exact-v0 TFBind10-Exact-v0"
experiments="bo_qei cma_es reinforce gradient_ascent ensemble_min ensemble_mean"

MAX_JOBS=4
AVAILABLE_GPUS="0"
MAX_RETRIES=0

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

for task in $tasks; do 
    for expt in $experiments; do
        for seed in {1..8}; do 

            check_jobs
            gpu_allocation=$(get_gpu_allocation $job_number)
            ((job_number++))
            run_with_retry "src/run_default.py \
                experiment=${expt} \
                ++task.task_name=${task} \
                ++seed=${seed}
                ++logger.job_type=1031_run" \
                "$gpu_allocation" & 
        done 
    done 
done 

wait