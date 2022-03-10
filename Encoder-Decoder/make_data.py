# Script to generate data
import os
from tqdm.auto import tqdm

base_path = "data/part1"
dev_path = os.path.join(base_path, "development_languages")
surprise_path = os.path.join(base_path, "surprise_languages")
test_path = os.path.join(base_path, "ground-truth")
save_path = "openmt_data"

def get_language_codes(path: str):
    codes = []
    for filename in os.listdir(path):
        if not filename.endswith('.train'):
            continue
    
        code = filename.split('.')[0].strip()
        codes.append(code)
    
    return codes


def get_source_target(path):
    with open(path) as df:
        source, target = [], []
        for line in df:
            line = line.strip()
            lemma, form, tag = line.split('\t')
            lemma_tokens = list(lemma.strip())
            form_tokens = list(form.strip())
            tag_tokens = tag.split(";")
            
            source.append(" ".join(lemma_tokens + tag_tokens) + '\n')
            target.append(" ".join(form_tokens) + '\n')
        
    return source, target


def make_data(path, code: str):
    train_file = os.path.join(path, code + ".train")
    dev_file = os.path.join(path, code + ".dev")
    test_file = os.path.join(test_path, code + ".test")
    
    train_source, train_target = get_source_target(train_file)
    dev_source, dev_target = get_source_target(dev_file)
    test_source, test_target = get_source_target(test_file) 
    
    os.makedirs(os.path.join(save_path, code), exist_ok=True)
    
    with open(os.path.join(save_path, code, "train.src"), "w") as tf:
        tf.writelines(train_source)
    
    with open(os.path.join(save_path, code, "train.trg"), "w") as tf:
        tf.writelines(train_target)
        
    with open(os.path.join(save_path, code, "dev.src"), "w") as df:
        df.writelines(dev_source)
    
    with open(os.path.join(save_path, code, "dev.trg"), "w") as df:
        df.writelines(dev_target)
        
    with open(os.path.join(save_path, code, "test.src"), "w") as tf:
        tf.writelines(test_source)
    
    with open(os.path.join(save_path, code, "test.trg"), "w") as tf:
        tf.writelines(test_target)


dev_codes = get_language_codes(dev_path)
surprise_codes = get_language_codes(surprise_path)

for code in tqdm(dev_codes):
    make_data(dev_path, code)

for code in tqdm(surprise_codes):
    make_data(surprise_path, code)
            
            
        



        
