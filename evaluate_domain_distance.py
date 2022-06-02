import numpy as np
import random



from scipy.stats import wasserstein_distance
from sklearn import svm

# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)




from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment





def wasserstein_distance_2d(Y1, Y2):

    num = np.shape(Y1)[0]


    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)

    dist = d[assignment].sum() / num

    return dist



def domain_distance(source, target):


    earth_move_dist = wasserstein_distance_2d(source, target)
    proxy_a_dist = proxy_a_distance(source, target)


    return earth_move_dist, proxy_a_dist



def domain_distance_ins(source, target, samples = 50):



    emd_list = []
    pad_list = []


    num_fg = len(source)

    for i in range(num_fg):
        src_i = source[i]
        tgt_i = target[i]

        random.shuffle(src_i)
        random.shuffle(tgt_i)

        src_selected = src_i[:samples]
        tgt_selected = tgt_i[:samples]

        src_selected_array = np.asarray(src_selected)
        tgt_selected_array = np.asarray(tgt_selected)

        emd_ci = wasserstein_distance_2d(src_selected_array, tgt_selected_array)
        pad_ci = proxy_a_distance(src_selected_array, tgt_selected_array)


        emd_list.append(emd_ci)
        pad_list.append(pad_ci)



    return emd_list, pad_list