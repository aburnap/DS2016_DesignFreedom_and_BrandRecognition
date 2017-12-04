import numpy as np

attr_list = ['Active', 'Aggressive','Distinctive','Expressive','Innovative','Luxurious','Powerful','Sporty','Well Proportioned','Youthful']

DESIGNS = [elm+str(num) for elm in ['a','b','c','l'] for num in range(0,5)]
morphed_DESIGNS = [elm+str(num) for elm in ['A','B','C','L'] for num in range(0,8)]
DESIGNS.extend(morphed_DESIGNS)


for attr_name in attr_list:
    print attr_name
    rank_file_name = '../data/processed_data/partial_rankings_from_mturk/' + attr_name + '_mturk_labeled.csv'
    rank_file =  np.genfromtxt(rank_file_name, delimiter=',',dtype=str)
    
    
    rank_file_id = np.empty(rank_file.shape)
    for car_id, car_name in enumerate(DESIGNS):
        rank_file_id[np.where(rank_file == car_name)] = car_id
    
    rank_file_id = rank_file_id.astype(int)
    save_name = '../data/processed_data/partial_rankings_from_mturk/partial_ranking_with_ID/' + attr_name + '_mturk_labeled_with_ID.csv'
    np.savetxt(save_name,rank_file_id,delimiter=',')