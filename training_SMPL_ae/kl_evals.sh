#!/bin/bash

DATASETS=("t2m" "xmo" "hml3dxmo" "idea400" "xmoI400" "hml3dI400" "hml3dxmoI400")
DATASETS_COMMAND=( "t2m" "xmo" )
LDS=(64)   

for DATASET in "${DATASETS_COMMAND[@]}"; do
  #echo $DATASET
  for IDEA in "${DATASETS[@]}"; do
    for LD in "${LDS[@]}"; do
      EXP_NAME="klvae_${IDEA}_ld${LD}"
      #echo $EXP_NAME
      CONFIG="experiments/${EXP_NAME}/config.yaml"
      CKPT="experiments/${EXP_NAME}/checkpoints/step_1000000.pth"

      if [[ -f "$CONFIG" && -f "$CKPT" ]]; then
        echo "Running eval for $EXP_NAME, model is evaluated on $DATASET"
        python eval.py --config "$CONFIG" --checkpoint "$CKPT" --dataset "$DATASET"
      else
        echo "Skipping $EXP_NAME (missing config or checkpoint)"
      fi
    done
  done
done


