PATTERN="2fwl_l"
# 2wl, 2wl_l, 2fwl, 2fwl_l

DATASET="Cora"
# Cora, Citeseer, Pubmed
# ogbl-ddi ?

SEED=0

python 2WLtest.py \
  --pattern $PATTERN \
  --dataset $DATASET \
  --device 1 \
  --seed $SEED \
  --subgraph "original"

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 0 \
#   --seed $SEED \
#   --subgraph cycle

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 2 \
#   --seed $SEED \
#   --subgraph incoming

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 3 \
#   --seed $SEED \
#   --subgraph outgoing

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 3 \
#   --seed $SEED \
#   --subgraph alpha