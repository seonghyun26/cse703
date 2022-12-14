PATTERN="2wl"
# 2wl, 2wl_l, 2fwl, 2fwl_l

DATASET="Cora"
# DATASET="jssp"
# Cora, Citeseer, Pubmed
# ogbl-ddi ?

SEED=0

python 2WLtest.py \
  --pattern $PATTERN \
  --dataset $DATASET \
  --device $1 \
  --seed $SEED \
  --name "jssp test" \
  --instanceSize "30_20"

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device $1 \
#   --seed $SEED \
#   --subgraph "1n2hop" \
#   --name "2wl_1n2hop"

# python 2WLtest.py \
  # --pattern $PATTERN \
  # --dataset $DATASET \
  # --device $1 \
  # --seed $SEED \
  # --subgraph "1n2hopMLP4" \
  # --name "2wl_1n2hopMLP_test4"

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device $1 \
#   --seed $SEED \
#   --subgraph "4hop" \
#   --name "2wl_4hop"
