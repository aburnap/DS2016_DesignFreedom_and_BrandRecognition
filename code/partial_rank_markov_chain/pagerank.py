#-----------------------------------------------------------------------------
#               Partial Rank Aggregation by Pagerank
#               Author: Yanxin Pan
#               Date: Dec 2, 2014
#               Notes: This is the code used for rank aggregation of attributes
#-----------------------------------------------------------------------------

from collections import Counter
import numpy as np
import numpy.linalg as LA


def compute_pageranks(google_mat, tol=1e-8):
    """ Compute the pagerank from the google matrix.

    Power method iterations are done until a tolerance of tol is achieved.
    """

    m, n = google_mat.shape
    if m != n:
        raise Exception('Expected first argument to be a square matrix')

    # the unif distribution
    dist = np.zeros(n) + 1.0 / n

    rel_change = 1.0
    while rel_change > tol:
        new_dist = dist.dot(google_mat)
        rel_change = LA.norm(new_dist - dist)/LA.norm(dist)
        dist = new_dist

    if not np.allclose(dist.dot(google_mat), dist):
        raise Exception('Power method did not find the stationary \
                        distribution')

    return dist

def adj_m1(rank_file,num_car):
    
    '''
    The rank_file should be a n*4 matrix, where n is the number of cars.
    Each row of this matrix represent a partial rank list for 4 cars.
    The elements is the id of cars.
    
    If the current state is car P, then the next state is chosen as follows: first pick a ranking
    uniformly from all the partial lists containing P, then pick a page Q uniformly from the set
    {Q|Q is not rank lower than P}
    
    This function is used to construct adjacency matrix from partial rank
    '''

    adj_mat = np.zeros((num_car,num_car))
    
    for car_ind in range(num_car):
        rank_with_the_car_id = np.where(rank_file == car_ind)
        num_rank = len(rank_with_the_car_id[0])
        higher_rank_car = []
        
        for rank_ind in range(num_rank):
            rank = rank_with_the_car_id[0][rank_ind]
            car_higher = rank_with_the_car_id[1][rank_ind]
            for car_higher_id in range(car_higher,4):
                higher_rank_car.append(rank_file[rank, car_higher_id])
            
        rank_count = Counter(higher_rank_car)
        
        for car in rank_count.keys():
            adj_mat[car_ind,car] = rank_count[car]
            
    adj_sum = adj_mat.sum(axis=1, keepdims=True)        
    adj_mat = adj_mat/np.where(adj_sum==0,1,adj_sum)
    
    return adj_mat
    
def adj_m2(rank_file,num_car):
    
    '''
    The rank_file should be a n*4 matrix, where n is the number of cars.
    Each row of this matrix represent a partial rank list for 4 cars.
    The elements is the id of cars.
    
    If the current state is car P, then the next state is chosen as follows: first pick a ranking
    uniformly from all the partial lists containing P, then uniformly pick a page that was ranked
    by the chosen ranking. if Q is rank higher than P, then go to Q,else stay in P.
    
    This function is used to construct adjacency matrix from partial rank
    '''

    adj_mat = np.zeros((num_car,num_car))
    
    for car_ind in range(num_car):
        rank_with_the_car_id = np.where(rank_file == car_ind)
        num_rank = len(rank_with_the_car_id[0])
        higher_rank_car = []
        
        for rank_ind in range(num_rank):
            rank = rank_with_the_car_id[0][rank_ind]
            car_higher = rank_with_the_car_id[1][rank_ind]
            for car_higher_id in range(car_higher,4):
                higher_rank_car.append(rank_file[rank, car_higher_id])
            higher_rank_car = higher_rank_car + [car_ind] * (4-car_higher)
            
        rank_count = Counter(higher_rank_car)
        
        for car in rank_count.keys():
            adj_mat[car_ind,car] = rank_count[car]
            
    adj_sum = adj_mat.sum(axis=1, keepdims=True)        
    adj_mat = adj_mat/np.where(adj_sum==0,1,adj_sum)
    
    return adj_mat
    
    
def transit_mc(rank_file, alpha, n, adj_m):
    
    '''
    This function is used to convert the adjacency matrix to 
    a stochastic, irreducible and aperiodic Markov matrix
    
    alpha controls the proportion of time the random suffer
    follow the hyperlink
    
    n is the number of cars
    '''
    if adj_m != "adj_m1" and adj_m != "adj_m2":
        raise Exception("Expected adj_m functon to be either 'adj_m1' or 'adj_m2'.")
    
    
    if adj_m == 'adj_m1':
        adj_mat = adj_m1(rank_file,n)
    else:
        adj_mat = adj_m2(rank_file,n)
    
    
    # construct 1d array of no. of outlinks
    out_links = np.sum(adj_mat, axis=1)

    # create 0-1 array with 1's for dangling nodes,
    # i.e. nodes without any out links.
    dangling = np.ones(n)
    nonzero_rows = out_links > 0
    dangling[nonzero_rows] = 0


    # make matrix stochastic
    rank_one_change = 1.0/n * np.outer(dangling, np.ones(n))
    adj_mat_stoch = adj_mat + rank_one_change

    # rows of adj_mat_stoch should all sum to 1
    assert np.allclose(np.sum(adj_mat_stoch, axis=1), np.ones(n))

    # create the "google" matrix by adding a random surfing term
    google_mat = alpha*adj_mat_stoch + (1-alpha)*np.ones((n, n))/n

    # rows of adj_mat_stoch should all sum to 1
    assert np.allclose(np.sum(google_mat, axis=1), np.ones(n))
    
    
    return google_mat
    
def compute_rank_eig(google_mat):
    
    w, V = LA.eig(google_mat.T)


    close_to_1 = np.array(np.nonzero(np.isclose(w, 1))).reshape(1)

    if close_to_1.size:
        ind_of_1 = close_to_1[0]
    else:
        raise Exception('Expected at least 1 eigenvalue to be 1, none found')

    # extract the corresponding eignevector
    pageranks_eig = np.squeeze(V[:, ind_of_1])
    pageranks_eig = np.real(pageranks_eig)
    pageranks_eig = pageranks_eig/sum(pageranks_eig)
    
    
    return pageranks_eig


def compare_two_ranks(attr_name,num_car,alpha, adj_m):
    
    rank_file_name = '../data/processed_data/partial_rankings_from_mturk/partial_ranking_with_ID/' + attr_name + '_mturk_labeled_with_ID.csv'
    rank_file = np.loadtxt(rank_file_name, delimiter=',')
    rank_file = rank_file.astype(int)
    unique_car = np.unique(rank_file)
    #num_car = unique_car.shape[0]
    google_mat = transit_mc(rank_file,alpha,num_car, adj_m)
    pageranks = compute_pageranks(google_mat)
    pageranks_eig = compute_rank_eig(google_mat)


    if np.allclose(pageranks, pageranks_eig):
        print 'agree'
        save_name = '../data/processed_data/attribute_values/'+ attr_name  +'_full_rank.csv'  
        np.savetxt(save_name,pageranks,delimiter=',')
        print pageranks
    else:
        print 'do not agree'


#if __name__ == '__main__':
#    main()
