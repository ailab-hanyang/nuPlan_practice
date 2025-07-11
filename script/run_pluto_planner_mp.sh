cwd=$(pwd)
# CKPT_ROOT="$cwd/checkpoints"
CKPT_ROOT="$cwd/checkpoints"

PLANNER=$1  
BUILDER=$2  
FILTER=$3  
CKPT=$4  
VIDEO_SAVE_DIR=$5  

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=5 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.2 \
    enable_simulation_progress_bar=true \
    experiment_uid="pluto_planner/$FILTER" \
    planner.pluto_planner.render=true \
    planner.pluto_planner.planner_ckpt="$CKPT" \
    +planner.pluto_planner.save_dir="$VIDEO_SAVE_DIR"