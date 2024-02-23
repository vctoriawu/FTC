#!/bin/bash

# use this script to SWEEP

<< SWEEP_SETUP :
# setup the sweep
SWEEP_SETUP
##################################################
#### CLIP model
CONFIG_YML="sweep.yml"

##### Step 1: Uncomment to setup the controller, and copy the SWEEP_ID that is returned
wandb sweep $CONFIG_YML

##################################################
##### Step 2: Paste the SWEEP_ID here
##################################################
#### CLIP Loss
#SWEEP_ID="rcl_stroke/as_tab/ynbsblx3"


##### Uncomment the line bellow to update the controller's config! After running an agent, this will not work anymore and a new controller needs to be setup
#wandb sweep --update $SWEEP_ID $CONFIG_YML


####################################################################################################################
########################## STEP 3: Run the Sweep Agents ######################################################
####################################################################################################################
<< ASTab_CLIP_Loss_Sweep :
# run the sweep agents
# Purang27, Feb 5, Sweep R2p1D network,
ASTab_CLIP_Loss_Sweep
CONFIG_YML="sweep.yml"
NAME="Sweep_CLIP_loss"
RUNNAME=$NAME"_00"
SAVE_DIR="logs/sweep/"$RUNNAME
#
##### First GPU, depending on sweep.py's wandb.agent's count, this is repeated that many times
#export CUDA_VISIBLE_DEVICES=0
#python sweep.py
#
#sleep 60
#
##### Second GPU (to run in parallel).
#export CUDA_VISIBLE_DEVICES=1
#nohup python sweep.py --config_path=$CONFIG_YML --run_name=$RUNNAME --save_dir=$SAVE_DIR --SWEEP_ID=$SWEEP_ID > "logs/"$NAME"_1".txt &
