conda activate tf14
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python bridges.py \
  --mode letters \
  --overlap \
  --model_name_fw 1558M \
  --model_name_bw 1558M \
  --batch_size 10 \
  --length 200 \
  --ngrams 5 \
  --write

