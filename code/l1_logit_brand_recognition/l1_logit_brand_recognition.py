
import time
import numpy as np
from sklearn import linear_model, cross_validation, preprocessing, svm
from sklearn.grid_search import GridSearchCV

DESIGNS = [elm+str(num) for elm in ['a','b','c','l'] for num in range(0,5)]
morphed_DESIGNS = [elm+str(num) for elm in ['A','B','C','L'] for num in range(0,8)]
DESIGNS.extend(morphed_DESIGNS)

ATTRIBUTES = ['Active', 'Aggressive','Distinctive','Expressive','Innovative','Luxurious','Powerful','Sporty','Well Proportioned','Youthful']

features = np.zeros( [len(DESIGNS) ,len(ATTRIBUTES)])

for ind, attr in enumerate(ATTRIBUTES):
    features[:, ind] = np.loadtxt("../../data/processed_data/attribute_values/"+attr+"_full_rank.csv", delimiter=',')

baseline_features = features[0:20, :]

scaled_baseline_features = preprocessing.scale(baseline_features, axis=0)

num_data, num_features = np.shape(baseline_features)

a=[0]*5
b=[1]*5
c=[2]*5
l=[3]*5

a.extend(b)
a.extend(c)
a.extend(l)

brand_labels = np.array(a)

#---------------- Classifier Setup -------------------------------------------
clf_dict = { 0: 'Log Reg L1',
             1: 'Log Reg L2'}#,
#             3: 'SGD w/ ElasticNet',
#             4: 'Linear SVM'}
#             4: 'Decision Tree',
             #5: 'Random Forest',
             #6: 'Linear SVM'}

num_classifiers = len(clf_dict)
clf_array = np.empty((num_classifiers, ), dtype=object)
num_jobs = 1 # 12 if running on Foveros

#---------- L1 Logistic Regression -----------------------------------
lr_l1_tuned_parameters = [{'C': [0.01, 0.1, 1.0, 10, 100, 1000]}]
lr_l1_clf = GridSearchCV(estimator=linear_model.LogisticRegression(penalty='l1'),
                         param_grid=lr_l1_tuned_parameters, cv=3,
                         scoring='accuracy', refit=True, n_jobs=num_jobs)

#---------- L2 Logistic Regression -----------------------------------
lr_l2_tuned_parameters = [{'C': [0.01, 0.1, 1.0, 10, 100, 1000]}]
lr_l2_clf = GridSearchCV(estimator=linear_model.LogisticRegression(penalty='l2'),
                         param_grid=lr_l2_tuned_parameters, cv=3,
                         scoring='accuracy', refit=True, n_jobs=num_jobs)

huber_tuned_parameters = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}]
huber_clf = GridSearchCV(estimator=linear_model.SGDClassifier(penalty='l2', loss='huber'),
                       param_grid = huber_tuned_parameters, cv=3, scoring='accuracy',
                       refit=True, n_jobs=1)

#---------- Linear SVM -----------------------------------------------
svm_tuned_parameters = [{'C': [0.01, 0.1, 1.0, 10, 100, 1000]}]
svm_clf = GridSearchCV(estimator=svm.LinearSVC(loss='l2'), #dual=False
        param_grid = svm_tuned_parameters, cv=3, scoring='accuracy',
        refit=True, n_jobs=num_jobs)


clf_array[0] = lr_l1_clf
clf_array[1] = lr_l2_clf
#clf_array[2] = huber_clf
#clf_array[3] = svm_clf

num_experiments = 3
test_size_per_split = 0.2
seed = 0
#---------------- Define Error and Time Matrices -----------------------------
clf_error_matrix = np.empty((num_classifiers, num_experiments))
clf_time_matrix = np.empty((num_classifiers, num_experiments))

#---------------- Split Dataset into Train/ Test -------------------------
cv_iterator = cross_validation.ShuffleSplit(num_data, 
        n_iter=num_experiments, test_size=test_size_per_split, 
        random_state=seed)

#---------------- Begin Multiple Experiments for Each Classifier ---------
for exp_index, (train_index, test_index) in enumerate(cv_iterator):

    train_x = baseline_features[train_index, :]
    train_t = brand_labels[train_index]
    test_x = baseline_features[test_index, :]
    test_t = brand_labels[test_index]

    #---------- Train, Test, and Time ------------------------------------
    for clf_index, clf in enumerate(clf_array):
#            if reg_index ==4:
#                continue
        start_time = time.time()
        clf.fit(train_x, train_t)
        clf_time_matrix[clf_index, exp_index] = time.time() - start_time
        clf_error_matrix[clf_index, exp_index] = clf.score(test_x, test_t)

train_t

train_x.shape

#train_x

test_t

clf_error_matrix

clf_array[0].score(train_x,train_t)

np.mean(clf_error_matrix[0])

clf_array[0].score(test_x,test_t)

clf_time_matrix

omega=clf_array[0].best_estimator_.coef_

np.savetxt("omega_30percent_brand_recognition.csv", omega, delimiter=',')

omega

clf_array[0].best_estimator_.coef_


