embedding="uicpr.ui.embed"
task="ui"

python3 ./evaluation/test_embeddings.py \
    --train "../dataset/Taobao/raw_sample.csv.pos.tsv" \
    --test "../dataset/Taobao/raw_sample.csv.test.tsv" \
    --remove_dup 1 --task $task --num_test 2000 \
    --embedding $embedding --topk 50 --worker 10

python3 ./evaluation/eval.py $embedding.$task.rec 10
python3 ./evaluation/eval.py $embedding.$task.rec 20