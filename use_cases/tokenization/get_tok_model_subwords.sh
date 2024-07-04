#!/bin/bash

for tok_model in google-bert/bert-base-multilingual-cased \
                 FacebookAI/xlm-roberta-large \
                 openai-community/gpt2 \
                 intfloat/multilingual-e5-large; do

  name=$(echo ${tok_model} | cut -d '/' -f2)
  echo "Working on ${name} ..."
  echo "verses;avg_subwords;glottocode" > "results/avg_subwords_${name}.csv"

  for filename in data/verses/*; do
    echo "Working on ${filename} ..."
    python get_num_subwords.py -t ${tok_model} -v "${filename}" -o "results/avg_subwords_${name}.csv"
  done

done