#-----------------------------------------------------------------------------
#
#       Extract Data code for IDETC 2015 project
#       Experiment #2 - Contained partial ranking for baseline cars and the
#       morphed cars as well
#
#-----------------------------------------------------------------------------

from dtc.models import *
import numpy as np

# Global Variables
ATTRIBUTES = [str(elm.positive_label) for elm in Attribute.objects.all()]
LOWER_TIME_CUTOFF = 10
UPPER_TIME_CUTOFF = 60
ACCURACY_THRESHOLD = 0.3
QUERIES_PER_USER = 8
RANKINGS_PER_QUERY = 4
NUM_DESIGN_VARIABLES = 4

def get_valid_users():
    user_all = User.objects.all()
    valid_users = []
    num_valid_users = 0
    for ind, user in enumerate(user_all):
        try:
            user.userprofile
            if len(user.ranking_set.all()) == QUERIES_PER_USER:
                num_valid_users += 1
                valid_users.append(user)
            else:
                print "invalid user"
        except:
            print "no userprofile for index " + str(ind)
    return valid_users

def get_user_accuracy(user):
    '''
    Assumes user is a valid user -- i.e., that they did all partial rankings
    Gets the brand recognition accuracy of the user
    Checks for baseline car only, not on morphed cars
    '''
    num = 0
    den = 0
    for ranking in user.ranking_set.all():
        partial_rank = str(ranking.ranked_design_ids).split(",")
        recog_brands = str(ranking.recognized_brands).split(",")
        for i, p in enumerate(partial_rank):
            # check only on baseline images (coded with lowercase, morphed coded with uppercase)
            if p[0] in ['a','b','c','l']:
                den += 1
                if p[0] == recog_brands[i]:
                    num += 1
    return float(num)/float(den)

def get_brand_recognitions(users_all):
    '''
    Assumes users_all is a vector of valid users
    Gets the overall brand recognition from all users
    '''
    audi_num = 0
    audi_den = 0
    bmw_num = 0
    bmw_den = 0
    cadillac_num = 0
    cadillac_den = 0
    lexus_num = 0
    lexus_den = 0    
    for user in users_all:
        for ranking in user.ranking_set.all():
            partial_rank = str(ranking.ranked_design_ids).split(",")
            recog_brands = str(ranking.recognized_brands).split(",")
            for i, p in enumerate(partial_rank):
                if p[0] == 'a':
                    audi_den += 1
                    if recog_brands[i] == 'a':
                        audi_num += 1
                elif p[0] == 'b':
                    bmw_den += 1
                    if recog_brands[i] == 'b':
                        bmw_num += 1
                elif p[0] == 'c':
                    cadillac_den += 1
                    if recog_brands[i] == 'c':
                        cadillac_num += 1
                elif p[0] == 'l':
                    lexus_den += 1
                    if recog_brands[i] == 'l':
                        lexus_num += 1
    full_crowd_acc = [float(audi_num)/float(audi_den), float(bmw_num)/float(bmw_den), float(cadillac_num)/float(cadillac_den), float(lexus_num)/float(lexus_den)]
    full_crowd_num = [audi_den, bmw_den, cadillac_den, lexus_den]

    audi_num = 0
    audi_den = 0
    bmw_num = 0
    bmw_den = 0
    cadillac_num = 0
    cadillac_den = 0
    lexus_num = 0
    lexus_den = 0    
    for user in users_all:
        if get_user_accuracy(user) < ACCURACY_THRESHOLD:
            continue
        for ranking in user.ranking_set.all():
            partial_rank = str(ranking.ranked_design_ids).split(",")
            recog_brands = str(ranking.recognized_brands).split(",")
            for i, p in enumerate(partial_rank):
                if p[0] == 'a':
                    audi_den += 1
                    if recog_brands[i] == 'a':
                        audi_num += 1
                elif p[0] == 'b':
                    bmw_den += 1
                    if recog_brands[i] == 'b':
                        bmw_num += 1
                elif p[0] == 'c':
                    cadillac_den += 1
                    if recog_brands[i] == 'c':
                        cadillac_num += 1
                elif p[0] == 'l':
                    lexus_den += 1
                    if recog_brands[i] == 'l':
                        lexus_num += 1
    expert_acc = [float(audi_num)/float(audi_den), float(bmw_num)/float(bmw_den), float(cadillac_num)/float(cadillac_den), float(lexus_num)/float(lexus_den)]
    expert_num = [audi_den, bmw_den, cadillac_den, lexus_den]

    return full_crowd_acc, full_crowd_num, expert_acc, expert_num 

def get_morphed_brand_recognitions(users_all, save_file=False):
    '''
    Assumes users_all is a vector of valid users
    Gets the overall brand recognition from all users
    '''
    MORPHED_DESIGNS = np.array([elm+str(num) for elm in ['A','B','C','L'] for num in range(0,8)])
    num_den = np.zeros([len(MORPHED_DESIGNS), 2])

    for user in users_all:
        if get_user_accuracy(user) < ACCURACY_THRESHOLD:
            continue
        for ranking in user.ranking_set.all():
            partial_rank = str(ranking.ranked_design_ids).split(",")
            recog_brands = str(ranking.recognized_brands).split(",")
            for i, p in enumerate(partial_rank):
                if p[0] in ['A', 'B', 'C', 'L']:
                    ind = np.argmax(np.array(MORPHED_DESIGNS)==p)
                    num_den[ind, 1] += 1
                    if recog_brands[i] == p[0].lower():
                        num_den[ind, 0] += 1

    morphed_design_brand_accuracies = np.divide(num_den[:,0], num_den[:,1])
    if save_file:
        np.savetxt("../data/processed_data/design_freedoms_and_brand_recognition_for_morphed_cars/morphed_car_brand_recognition_baseline_acc_30.csv", morphed_design_brand_accuracies, delimiter=",")
    return morphed_design_brand_accuracies


def get_ranking_for_attribute(attribute_name):
    '''
    Input a string for the attribute name
    Output the full set of partial rankings for that attribute only for users that were valid
    meaning did all rankings, and only if they had higher that ACCURACY_THRESHOLD for
    brand recognition on baseline images
    '''
    valid_users = get_valid_users()
    attribute_ranks = []
    times = []
    for user in valid_users:
        if get_user_accuracy(user) < ACCURACY_THRESHOLD:
            continue
        user_attribute = str(user.userprofile.attribute.positive_label)
        if user_attribute == attribute_name:
            if user.userprofile.rank_order:
                user_ranking = [str(elm.ranked_design_ids).split(",") for elm in user.ranking_set.all()]
            else:
                user_ranking = [str(elm.ranked_design_ids).split(",")[::-1] for elm in user.ranking_set.all()]
            try:
                user_ranking = np.array(user_ranking).reshape([QUERIES_PER_USER, RANKINGS_PER_QUERY])
            except ValueError:
                print user.ranking_set.all()
                print user.pk
            user_times = [elm.completion_time for elm in user.ranking_set.all()]
            #try:
                #np.concatenate((attribute_ranks, user_ranking), axis=0)
            #except: 
            attribute_ranks.append(user_ranking)
            times.append(user_times)
    attribute_ranks = np.array(attribute_ranks)
    attribute_times = np.array(times)
    try:
        attribute_ranks = attribute_ranks.reshape([attribute_ranks.shape[0]*attribute_ranks.shape[1], RANKINGS_PER_QUERY])
    except IndexError:
        print attribute_ranks
        print attribute_name
    attribute_times = attribute_times.reshape([attribute_times.shape[0]*attribute_times.shape[1], 1])
    return attribute_ranks.astype(str), attribute_times.astype(float)

def save_attribute_data():
    for attribute in ATTRIBUTES:
        data_matrix, times = get_ranking_for_attribute(attribute)
        #filter_inds = np.where( (times > LOWER_TIME_CUTOFF) & (times < UPPER_TIME_CUTOFF) )[0]
        #filtered_data_matrix = data_matrix[filter_inds]
        #np.savetxt("../data/processed_data/mturk_labeled_12_17_14/"+attribute+"_mturk_labeled.csv", filtered_data_matrix, fmt="%i", delimiter=',')
        np.savetxt("../data/processed_data/partial_rankings_from_mturk/"+attribute+"_mturk_labeled.csv", data_matrix, fmt="%s", delimiter=',')





