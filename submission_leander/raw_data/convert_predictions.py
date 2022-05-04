import os

data_path = "../../Prefix-Suffix-Rule-Classification/development_languages"


if __name__ == '__main__':
    for prediction_file in sorted(os.listdir('.')):
        if not prediction_file.endswith('tsv'):
            continue
        
        lang = prediction_file.split('_')[0]
        code = prediction_file.split('.')[0]

        if 'dev' in prediction_file:
            data_file = lang + '.dev'
            test = False
        else:
            data_file = lang + '.test'
            test = True
            
        with open(os.path.join(data_path, data_file)) as df:
            data = [line.strip() for line in df]
        
        with open(prediction_file) as pf:
            predictions = []
            for i, line in enumerate(pf):
                if i == 0:
                    continue
                
                prediction = line.strip().split('\t')[0].split(" ")
                converted_prediction = []
                for char in prediction:
                    if ':' in char:
                        char = char.split(':')[1]
                    converted_prediction.append(char.strip())
                
                prediction = "".join(converted_prediction).replace('#', ' ')
                predictions.append(prediction)
        
        os.makedirs('converted_predictions', exist_ok=True)
        with open(f'converted_predictions/{code}.{"dev" if not test else "test"}', 'w') as sf:
            for prediction, dp in zip(predictions, data):
                if test:
                    lemma, tags = dp.split('\t')
                    sf.write(f"{lemma}\t{prediction}\t{tags}\n")
                else:
                    lemma, target, tags = dp.split('\t')
                    sf.write(f"{lemma}\t{prediction}\t{target}\t{tags}\n")
