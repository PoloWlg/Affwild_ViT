import pickle 


path = '/projets/AS84330/Datasets/Abaw6_EXPR_no_preprocessing/dataset_info_-1.pkl'
with open(path, 'rb') as handle:
        data = pickle.load(handle)
        
        length_train = 0
        length_val = 0
        for index, partition in enumerate(data['partition']):
            if partition == 'train':
                length_train += data['length'][index]
            elif partition == 'validate':
                length_val += data['length'][index]
                
        print(f"Length train: {length_train}")
        print(f"Length val: {length_val}")
        print(f"total length: {length_train + length_val}")