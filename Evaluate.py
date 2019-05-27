import numpy as np
from math import lgamma, pi, log1p
from scipy.stats import t


Record = np.dtype({
    'names': ['longitude', 'latitude', 'time', 'weekend', 'POI', 'cat6', 'cat16', 'cat54', 'POI_lon', 'POI_lat', 'region'],
    'formats': ['f', 'f', 'f', 'b', 'S32', 'i', 'i', 'i', 'f', 'f', 'i']})


def distance(lon1, lat1, lon2, lat2):
    lon1 = lon1 / 180 * np.pi
    lat1 = lat1 / 180 * np.pi
    lon2 = lon2 / 180 * np.pi
    lat2 = lat2 / 180 * np.pi
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(h)) * 6371000


def multivariate_t(x, mu, Sigma, df):
    p = Sigma.shape[0]
    dec = np.linalg.cholesky(Sigma)
    R_x_m = np.linalg.solve(dec, np.matrix.transpose(x)-mu)
    rss = np.power(R_x_m,2).sum(axis=0)
    logretval = lgamma(1.0*(p + df)/2) - (lgamma(1.0*df/2) + np.sum(np.log(dec.diagonal())) + p/2 * np.log(pi * df)) - 0.5 * (df + p) * log1p(rss/df)
    return np.exp(logretval)


def evaluate(testdata, region_num, user, POI_location, POI_cat):
    n_rc = np.load('parameters/n_rc.npy')
    n_ct = np.load('parameters/n_ct.npy')
    mu = np.load('parameters/mu.npy')
    Sigma = np.load('parameters/SSigma.npy')
    nu = np.load('parameters/nu.npy')
    sigma = np.load('parameters/sigma.npy')
    n_cz = np.load('parameters/n_cz.npy')
    Phi_ur = np.load('parameters/Phi_ur'+str(user-1)+'.npy')
    n_r = np.sum(n_rc, axis=1)

    period_num = 2
    b = 3.0
    beta = 10.0
    gamma = 10.0
    tau_0 = 2
    rho_0 = 10
    count_true = 0
    count_false = 0
    count_top2 = 0
    count_top3 = 0
    count_true_bayes = 0
    count_top2_bayes = 0
    count_top3_bayes = 0
    count_true_dist = 0
    count_top2_dist = 0
    count_top3_dist = 0
    only = 0
    dist_sum = 0

    longitude = testdata['longitude']
    latitude = testdata['latitude']
    location = np.array([longitude, latitude])
    time = testdata['time']

    prob_max = 0
    POI_pred = "None"
    prob_second = 0
    POI_second = "None"
    prob_third = 0
    POI_third = "None"
    prob_max_bayes = 0
    POI_bayes = "None"
    prob_second_bayes = 0
    POI_second_bayes = "None"
    prob_third_bayes = 0
    POI_third_bayes = "None"
    prob_max_dist = 0
    POI_dist = "None"
    prob_second_dist = 0
    POI_second_dist = "None"
    prob_third_dist = 0
    POI_third_dist = "None"

    p_r_max = 0
    for r in range(region_num):
        p_r = multivariate_t(location, mu[r], Sigma[r], n_r[r] + rho_0 - 1)
        if p_r > p_r_max:
            p_r_max = p_r
            r_max = r
    for POI, location in POI_location.items():
        p_r_sum = 0
        dist = distance(longitude, latitude, location[0], location[1])
        if dist < 500:
            only += 1
            POI_c = POI_cat[POI]
            p_time = 0
            for z in range(period_num):
                p_time += (n_cz[POI_c, z] + gamma) / sum(n_cz[POI_c, :] + gamma) * t.pdf((time - nu[POI_c, z]) / np.sqrt(sigma[POI_c, z]), n_cz[POI_c, z] + tau_0)
            p_dist = np.exp(dist * dist / (-10000))
            # for r in range(region_num):
            #     p_r = multivariate_t(np.array([location[0], location[1]]), mu[r], Sigma[r], n_r[r] + rho_0 - 1)
            #     p_cat = Phi_ur[r, POI_c] / sum(Phi_ur[r, :])
            #     p_r_sum += p_r * p_cat
            p_cat = Phi_ur[r_max, POI_c] / sum(Phi_ur[r_max, :])
            prob = p_time * p_dist * p_cat
            
            prob_bayes = n_ct[POI_c, int(time)] / sum(n_ct[:, int(time)]) / sum(n_rc[:, POI_c])
            prob_bayes *= np.power(dist, -1.0)  # Beijing: 1.0; Guangzhou: 1.1
            prob_dist = 1 / dist

            if prob > prob_max:
                prob_third = prob_second
                POI_third = POI_second
                prob_second = prob_max
                POI_second = POI_pred
                prob_max = prob
                POI_pred = POI
            elif prob > prob_second:
                prob_third = prob_second
                POI_third = POI_second
                prob_second = prob
                POI_second = POI
            elif prob > prob_third:
                prob_third = prob
                POI_third = POI

            if prob_bayes > prob_max_bayes:
                prob_third_bayes = prob_second_bayes
                POI_third_bayes = POI_second_bayes
                prob_second_bayes = prob_max_bayes
                POI_second_bayes = POI_bayes
                prob_max_bayes = prob_bayes
                POI_bayes = POI
            elif prob_bayes > prob_second_bayes:
                prob_third_bayes = prob_second_bayes
                POI_third_bayes = POI_second_bayes
                prob_second_bayes = prob_bayes
                POI_second_bayes = POI
            elif prob_bayes > prob_third_bayes:
                prob_third_bayes = prob_bayes
                POI_third_bayes = POI

            if prob_dist > prob_max_dist:
                prob_third_dist = prob_second_dist
                POI_third_dist = POI_second_dist
                prob_second_dist = prob_max_dist
                POI_second_dist = POI_dist
                prob_max_dist = prob_dist
                POI_dist = POI
            elif prob_dist > prob_second_dist:
                prob_third_dist = prob_second_dist
                POI_third_dist = POI_second_dist
                prob_second_dist = prob_dist
                POI_second_dist = POI
            elif prob_dist > prob_third_dist:
                prob_third_dist = prob_dist
                POI_third_dist = POI

    if POI_pred == testdata['POI']:
        count_true += 1
        count_top2 += 1
        count_top3 += 1
    elif POI_second == testdata['POI']:
        count_top2 += 1
        count_top3 += 1
        count_false += 1
    elif POI_third == testdata['POI']:
        count_top3 += 1
        count_false += 1
    else:
        count_false += 1

    if POI_bayes == testdata['POI']:
        count_true_bayes += 1
        count_top2_bayes += 1
        count_top3_bayes += 1
    elif POI_second_bayes == testdata['POI']:
        count_top2_bayes += 1
        count_top3_bayes += 1
    elif POI_third_bayes == testdata['POI']:
        count_top3_bayes += 1

    if POI_dist == testdata['POI']:
        count_true_dist += 1
        count_top2_dist += 1
        count_top3_dist += 1
    elif POI_second_dist == testdata['POI']:
        count_top2_dist += 1
        count_top3_dist += 1
    elif POI_third_dist == testdata['POI']:
        count_top3_dist += 1

    return count_true, count_top2, count_top3, count_false, count_true_bayes, count_top2_bayes, count_top3_bayes, count_true_dist, count_top2_dist, count_top3_dist


def main():
    city = 'Guangzhou'
    category_num = 16
    region_num = 225
    if city == 'Guangzhou':
        User_num = 427
    elif city == 'Beijing':
        User_num = 390

    count_true = 0
    count_false = 0
    count_top2 = 0
    count_top3 = 0
    count_true_bayes = 0
    count_top2_bayes = 0
    count_top3_bayes = 0
    count_true_dist = 0
    count_top2_dist = 0
    count_top3_dist = 0

    testset = np.load(city + '_dataset/testset.npy')
    f = open(city + '_dataset/dict_POI_location.txt', 'r')
    a = f.read()
    POI_location = eval(a)
    f.close()
    f = open(city + '_dataset/dict_POI_cat' + str(category_num) + '.txt', 'r')
    a = f.read()
    POI_cat = eval(a)
    f.close()

    for user in range(User_num):
        print('User' + str(user+1))
        [true1, true2, true3, false, b_true1, b_true2, b_true3, d_true1, d_true2, d_true3] = evaluate(testset[user], region_num, user+1, POI_location, POI_cat)

        count_true += true1
        count_top2 += true2
        count_top3 += true3
        count_true_bayes += b_true1
        count_top2_bayes += b_true2
        count_top3_bayes += b_true3
        count_true_dist += d_true1
        count_top2_dist += d_true2
        count_top3_dist += d_true3
        count_false += false

    print("True:%d\tFalse:%d\tAcc:%f" % (count_true, count_false, count_true/(count_true+count_false)))
    print("TOP2 Acc:%f\tTOP3 Acc:%f" % (count_top2/(count_true+count_false), count_top3/(count_true+count_false)))
    print("BayesTrue:%d\tBayesAcc:%f" % (count_true_bayes, count_true_bayes/(count_true+count_false)))
    print("BayesTOP2 Acc:%f\tBayesTOP3 Acc:%f" % (count_top2_bayes/(count_true+count_false), count_top3_bayes/(count_true+count_false)))
    print("DistTrue:%d\tDistAcc:%f" % (count_true_dist, count_true_dist/(count_true+count_false)))
    print("DistTOP2 Acc:%f\tDistTOP3 Acc:%f" % (count_top2_dist/(count_true+count_false), count_top3_dist/(count_true+count_false)))


if __name__ == '__main__':
    main()
