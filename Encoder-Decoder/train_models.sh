mkdir config
mkdir predictions

python make_data.py
python make_configs.py

for lang in $(ls opennmt_data); do
    onmt_build_vocab -config "config/${lang}.yaml" -n_sample 1000000 ;
    onmt_train -config "config/${lang}.yaml" ;
    models=(./openmt_models/${lang}/*.pt) ;
    onmt_translate -model "${models[0]}" -src "opennmt_data/${lang}/test.src" -output "predictions/${lang}.pred" -gpu 0 ;
done ;
    
