import numpy as np
from math import lgamma, pi, log1p
from scipy.stats import t
import matplotlib.pyplot as plt
import scipy.io as sio


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
    logretval = lgamma(1.0*(p + df)/2) - (lgamma(1.0*df/2) + np.sum(np.log(dec.diagonal()))
       + p/2 * np.log(pi * df)) - 0.5 * (df + p) * log1p((rss/df) )
    return (np.exp(logretval))


def blockread(buf, sep):
    while sep in buf:
        pos = buf.index(sep)
        yield buf[:pos]
        buf = buf[pos + len(sep):]
    yield buf


def evaluate(dataset, testdata, region_num, category_num, user, city, POI_location, POI_cat, POI_name):
    f = open(city + '_dataset/userdata/candidatelist'+str(user)+'.txt', 'r', encoding='UTF-8')
    a = f.read()
    candidatelist = eval(a)
    f.close()
    n_rc = np.load('parameters/n_rc.npy')
    n_ct = np.load('parameters/n_ct.npy')
    mu = np.load('parameters/mu.npy')
    Sigma = np.load('parameters/SSigma.npy')
    nu = np.load('parameters/nu.npy')
    sigma = np.load('parameters/sigma.npy')
    n_cz = np.load('parameters/n_cz.npy')
    Phi_ur = np.load('parameters/Phi_ur'+str(user-1)+'.npy')
    n_r = np.sum(n_rc, axis=1)
    record_num = dataset.size
    lon_list = np.zeros(record_num, dtype=np.float64)
    lat_list = np.zeros(record_num, dtype=np.float64)
    cat_count = np.zeros(category_num, dtype=np.float64)

    period_num = 2
    b = 3.0
    beta = 10.0
    gamma = 10.0
    tau_0 = 2
    rho_0 = 10
    count_true = 0
    count_false = 0
    count_true_top2 = 0
    count_true_top3 = 0
    baseline_true = 0
    count_true_top2_baseline = 0
    count_true_top3_baseline = 0
    dist_true = 0
    count_true_top2_dist = 0
    count_true_top3_dist = 0
    count = -1
    only = 0
    dist_sum = 0
    trace_not_found = 0
    f.close()

    for d in dataset:
        count += 1
        c_i = d['cat'+str(category_num)]
        longitude = d['longitude']
        latitude = d['latitude']
        location = np.array([longitude, latitude])
        time = d['time']
        prob_max = 0
        POI_pred = "None"
        lon_list[count] = longitude
        lat_list[count] = latitude
        cat_count[c_i] += 1

        p_r_max = 0
        for r in range(region_num):
            p_r = multivariate_t(location, mu[r], Sigma[r], n_r[r] + rho_0 - 1)
            if p_r > p_r_max:
                p_r_max = p_r
                r_i = r
        Phi_ru_temp = Phi_ur[r_i, :]
        for POI in candidatelist[count+1]:
            dist = distance(longitude, latitude, location[0], location[1])
            POI_c = POI_cat[POI]
            p_time = 0
            for z in range(period_num):
                p_time += (n_cz[POI_c, z] + gamma) / sum(n_cz[POI_c, :] + gamma) * t.pdf((time - nu[POI_c, z]) / np.sqrt(sigma[POI_c, z]), n_cz[POI_c, z] + tau_0)
            p_dist = np.exp(dist * dist / (-10000))
            p_cat = Phi_ru_temp[POI_c] / sum(Phi_ru_temp)
            prob = p_time * p_dist * p_cat
            if prob > prob_max:
                prob_max = prob
                POI_pred = POI
                # print("%s:p_time:%f,p_dist:%f,p_cat:%f,p:%f" % (POI_name[POI], p_time, p_dist, p_cat, prob))
        if POI_pred == "None":
            # print("Trace Not Found")
            trace_not_found += 1
        else:
            # print("Pred:%s" % POI_name[POI_pred])
            Phi_ur[r_i, POI_cat[POI_pred]] += 1

    c_i = testdata['cat'+str(category_num)]
    longitude = testdata['longitude']
    latitude = testdata['latitude']
    location = np.array([longitude, latitude])
    time = testdata['time']
    dist_sum += distance(longitude, latitude, testdata['POI_lon'], testdata['POI_lat'])

    prob_max = 0
    POI_pred = "None"
    prob_second = 0
    POI_second = "None"
    prob_third = 0
    POI_third = "None"
    prob_max_baseline = 0
    POI_baseline = "None"
    prob_second_baseline = 0
    POI_second_baseline = "None"
    prob_third_baseline = 0
    POI_third_baseline = "None"
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
            r_i = r
    print("Cat:%d\tTime:%f\tRegion:%d" % (POI_cat[testdata['POI']], time, r_i))
    for POI, location in POI_location.items():
        dist = distance(longitude, latitude, location[0], location[1])
        if dist < 500:
            only += 1
            POI_c = POI_cat[POI]
            p_time = 0
            for z in range(period_num):
                p_time += (n_cz[POI_c, z] + gamma) / sum(n_cz[POI_c, :] + gamma) * t.pdf((time - nu[POI_c, z]) / np.sqrt(sigma[POI_c, z]), n_cz[POI_c, z] + tau_0)
            p_dist = np.exp(dist * dist / (-10000))
            p_cat = Phi_ur[r_i, POI_c] / sum(Phi_ur[r_i, :])
            prob = p_time * p_dist * p_cat
            prob_baseline = n_ct[POI_c, int(time)] / sum(n_ct[:, int(time)]) / sum(n_rc[:, POI_c])
            prob_baseline *= np.power(dist, -1.0)  # Beijing1.0; Guangzhou1.1
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

            if prob_baseline > prob_max_baseline:
                prob_third_baseline = prob_second_baseline
                POI_third_baseline = POI_second_baseline
                prob_second_baseline = prob_max_baseline
                POI_second_baseline = POI_baseline
                prob_max_baseline = prob_baseline
                POI_baseline = POI
            elif prob_baseline > prob_second_baseline:
                prob_third_baseline = prob_second_baseline
                POI_third_baseline = POI_second_baseline
                prob_second_baseline = prob_baseline
                POI_second_baseline = POI
            elif prob_baseline > prob_third_baseline:
                prob_third_baseline = prob_baseline
                POI_third_baseline = POI

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
            # print("%s:\tp_time:%f\tp_dist:%f\tp_cat:%f\tp:%f" % (POI_name[POI], p_time, p_dist, p_cat, prob))
    # if POI_pred == "None":
        # print("Test POI Not Found")
    # else:
        # print("True:%s\tPred:%s" % (POI_name[testdata['POI']], POI_name[POI_pred]))
    if POI_pred == testdata['POI']:
        count_true += 1
        count_true_top2 += 1
        count_true_top3 += 1
    elif POI_second == testdata['POI']:
        count_true_top2 += 1
        count_true_top3 += 1
        count_false += 1
    elif POI_third == testdata['POI']:
        count_true_top3 += 1
        count_false += 1
    else:
        count_false += 1

    if POI_baseline == testdata['POI']:
        baseline_true += 1
        count_true_top2_baseline += 1
        count_true_top3_baseline += 1
    elif POI_second_baseline == testdata['POI']:
        count_true_top2_baseline += 1
        count_true_top3_baseline += 1
    elif POI_third_baseline == testdata['POI']:
        count_true_top3_baseline += 1

    if POI_dist == testdata['POI']:
        dist_true += 1
        count_true_top2_dist += 1
        count_true_top3_dist += 1
    elif POI_second_dist == testdata['POI']:
        count_true_top2_dist += 1
        count_true_top3_dist += 1
    elif POI_third_dist == testdata['POI']:
        count_true_top3_dist += 1

    return count_true, count_true_top2, count_true_top3, count_false, baseline_true, count_true_top2_baseline, count_true_top3_baseline, dist_true, count_true_top2_dist, count_true_top3_dist


def main():
    category_num = 16
    # city = 'Beijing'
    city = 'Guangzhou'
    if city == 'Guangzhou':
        User_num = 427
    elif city == 'Beijing':
        User_num = 390
    region_num = 225

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

    testdata = np.load(city + '_dataset/testdata.npy')
    f = open(city + '_dataset/dict_POI_location.txt', 'r')
    a = f.read()
    POI_location = eval(a)
    f.close()
    f = open(city + '_dataset/dict_POI_cat' + str(category_num) + '.txt', 'r')
    a = f.read()
    POI_cat = eval(a)
    f.close()
    f = open(city + '_dataset/dict_POI_name.txt', 'r', encoding='UTF-8')
    a = f.read()
    POI_name = eval(a)
    f.close()

    for user in range(User_num):
        print(user+1)
        dataset = np.load(city + '_dataset/userdata/usertrace_'+str(user+1)+'.npy')
        trace_num = dataset.size

        [true1, true2, true3, false, b_true1, b_true2, b_true3, d_true1, d_true2, d_true3] = evaluate(dataset, testdata[user], region_num, category_num, user+1, city, POI_location, POI_cat, POI_name)

        count_true += true1
        count_false += false
        count_top2 += true2
        count_top3 += true3
        count_true_bayes += b_true1
        count_top2_bayes += b_true2
        count_top3_bayes += b_true3
        count_true_dist += d_true1
        count_top2_dist += d_true2
        count_top3_dist += d_true3

    print("True:%d\tFalse:%d\tAcc:%f" % (count_true, count_false, count_true/(count_true+count_false)))
    print("TOP2 Acc:%f\tTOP3 Acc:%f" % (count_top2/(count_true+count_false), count_top3/(count_true+count_false)))
    print("BayesTrue:%d\tBayesAcc:%f" % (count_true_bayes, count_true_bayes/(count_true+count_false)))
    print("BayesTOP2 Acc:%f\tBayesTOP3 Acc:%f" % (count_top2_bayes/(count_true+count_false), count_top3_bayes/(count_true+count_false)))
    print("DistTrue:%d\tDistAcc:%f" % (count_true_dist, count_true_dist/(count_true+count_false)))
    print("DistTOP2 Acc:%f\tDistTOP3 Acc:%f" % (count_top2_dist/(count_true+count_false), count_top3_dist/(count_true+count_false)))


if __name__ == '__main__':
    main()
