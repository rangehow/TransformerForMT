corpus=/home/ruanjunhao/translate
main=${corpus}/iwslt14.tokenized.de-en

if [ ! -d "$main" ]; then
  echo "语料文件夹加载失败"
  exit
fi
echo $main

#src指向德文，tgt指向英文，然后对应生成两个vocab，和一个合并的文件
python3 prepare.py --train_data_src ${main}/train.de --train_data_tgt ${main}/train.en --vocab_src ${main}/output/src.vocab --vocab_tgt ${main}/output/tgt.vocab --output_file ${main}/output/train.txt
