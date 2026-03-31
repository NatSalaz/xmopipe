#!/bin/bash

DATASETS=("t2m" "xmo" "hml3dxmo" "idea400" "xmoI400" "hml3dI400" "hml3dxmoI400")
DATASETS_COMMAND=("t2m" "xmo" "hml3dxmo" "idea400")
CBS=(512 1024)
LDS=(512)   

for DATASET in "${DATASETS_COMMAND[@]}"; do
  echo $DATASET
  for IDEA in "${DATASETS[@]}"; do
    for LD in "${LDS[@]}"; do
      for CB in "${CBS[@]}"; do
        EXP_NAME="rvqvae_${IDEA}_ld${LD}_cb${CB}"
        echo $EXP_NAME
        CONFIG="experiments/${EXP_NAME}/config.yaml"
        CKPT="experiments/${EXP_NAME}/checkpoints/best.pth"
        if [[ -f "$CONFIG" && -f "$CKPT" ]]; then
          echo "Running eval for $EXP_NAME"
          python eval.py --config "$CONFIG" --checkpoint "$CKPT" --dataset "$DATASET"
        else
          echo "Skipping $EXP_NAME (missing config or checkpoint)"
        fi
      done
    done
  done
done


