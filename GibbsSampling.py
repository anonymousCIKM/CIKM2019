import numpy as np
import time as timer
from math import lgamma, pi, log1p
from scipy.stats import invgamma, invwishart, multivariate_normal, norm, t

Record_sample = np.dtype({
    'names': ['longitude', 'latitude', 'time', 'weekend', 'POI', 'cat6', 'cat16', 'cat54', 'r', 'c', 'z', 'u'],
    'formats': ['f', 'f', 'f', 'b', 'S32', 'i', 'i', 'i', 'i', 'i', 'i', 'i']})


def dataloader(city):
    data = np.load(city + '_dataset/labeled.npy')
    dataset = np.array([], dtype=Record_sample)
    for d in data:
        dataset = np.concatenate([dataset,
                                  np.array([(d['longitude'], d['latitude'], d['time'], d['weekend'], d['POI'],
                                             d['cat6'], d['cat16'], d['cat54'], -1, -1, -1, d['user'])],
                                           dtype=Record_sample)])
    return dataset


def multivariate_t(x, mu, Sigma, df):
    p = Sigma.shape[0]
    dec = np.linalg.cholesky(Sigma)
    R_x_m = np.linalg.solve(dec, np.matrix.transpose(x)-mu)
    rss = np.power(R_x_m,2).sum(axis=0)
    logretval = lgamma(1.0*(p + df)/2) - (lgamma(1.0*df/2) + np.sum(np.log(dec.diagonal())) + p/2 * np.log(pi * df)) - 0.5 * (df + p) * log1p(rss/df)
    return np.exp(logretval)


def main():
    # city = 'Beijing'
    city = 'Guangzhou'
    category_num = 16
    region_num = 225
    if city == 'Guangzhou':
        user_num = 427
    elif city == 'Beijing':
        user_num = 390
    record = dataloader(city)
    longitude = record['longitude']
    latitude = record['latitude']
    time = record['time']
    print(city + ' dataset loaded. Record number: %d.' % record.size)
    gibbs_iter_num = 30

    alpha = 10000.0
    beta = 10.0
    gamma = 10.0
    upsilon_0 = np.array([np.mean(longitude), np.mean(latitude)])
    upsilon_0_mat = upsilon_0[:, np.newaxis]
    kappa_0 = 1.0
    rho_0 = 10.0
    zeta_0 = np.cov(longitude, latitude)
    epsilon_0 = np.mean(time)
    lambda_0 = 1.0
    tau_0 = 2.0
    xi_0 = np.var(time)
    b = 3.0
    period_num = 2

    Sigma = np.zeros((region_num, 2, 2), dtype=np.float64)
    Sigma_sum = np.zeros((region_num, 2, 2), dtype=np.float64)
    mu = np.zeros((region_num, 2), dtype=np.float64)
    mu_sum = np.zeros((region_num, 2), dtype=np.float64)
    sigma = np.zeros((category_num, period_num), dtype=np.float64)
    sigma_sum = np.zeros((category_num, period_num), dtype=np.float64)
    nu = np.zeros((category_num, period_num), dtype=np.float64)
    nu_sum = np.zeros((category_num, period_num), dtype=np.float64)
    for r in range(region_num):
        Sigma[r] = invwishart.rvs(rho_0, zeta_0)
        mu[r] = np.random.multivariate_normal(upsilon_0, Sigma[r]/kappa_0)
    for c in range(category_num):
        for z in range(period_num):
            sigma[c, z] = invgamma.rvs(tau_0, xi_0)
            nu[c, z] = np.random.normal(epsilon_0, np.sqrt(sigma[c, z]/lambda_0))

    n_r = np.zeros(region_num, dtype=int)
    n_rc = np.zeros((region_num, category_num), dtype=int)
    n_c = np.zeros(category_num, dtype=int)
    n_cz = np.zeros((category_num, period_num), dtype=int)
    n_urc = np.zeros((user_num, region_num, category_num), dtype=int)

    for gibbs_iter in range(gibbs_iter_num):
        time_start = timer.time()
        print('Gibbs iteration %d' % gibbs_iter)
        np.random.shuffle(record)
        p_l = np.zeros(region_num, dtype=np.float64)
        i = 0
        for d in record:
            time = d['time']
            location = np.array([d['longitude'], d['latitude']])
            l_mat = location[:, np.newaxis]
            if gibbs_iter > 0:
                r_i = d['r']
                c_i = d['c']
                z_i = d['z']
                n_r[r_i] -= 1
                n_rc[r_i, c_i] -= 1
                n_urc[d['u'] - 1, r_i, c_i] -= 1
                n_c[c_i] -= 1
                n_cz[c_i, z_i] -= 1

                if n_r[r_i] > 0:
                    n = n_r[r_i]
                    mu_sum[r_i] -= location
                    mu_sum_mat = mu_sum[r_i]
                    mu_sum_mat = mu_sum_mat[:, np.newaxis]
                    Sigma_sum[r_i] -= l_mat*np.transpose(l_mat)
                    kappa = kappa_0 + n
                    mu[r_i] = (kappa_0*upsilon_0 + mu_sum[r_i]) / kappa
                    zeta = zeta_0 + kappa_0*n / kappa * (
                            (mu_sum_mat / n - upsilon_0_mat)*np.transpose(mu_sum_mat/n-upsilon_0_mat)) + (
                            Sigma_sum[r_i] - mu_sum_mat*np.transpose(mu_sum_mat)/n)
                    if ~np.all(np.linalg.eigvals(zeta) > 0):
                        zeta = zeta_0
                    rho = rho_0 + n
                    Sigma[r_i] = zeta * (kappa + 1) / kappa / (rho - 1)
                else:
                    Sigma[r_i] = invwishart.rvs(rho_0, zeta_0)
                    mu[r_i] = np.random.multivariate_normal(upsilon_0, Sigma[r_i]/kappa_0)
                    mu_sum[r_i] = 0
                    Sigma_sum[r_i] = np.zeros((2, 2), dtype=np.float64)

                if n_cz[c_i, z_i] > 0:
                    n = n_cz[c_i, z_i]
                    nu_sum[c_i, z_i] -= time
                    sigma_sum[c_i, z_i] -= time ** 2
                    lamb = lambda_0 + n
                    nu[c_i, z_i] = ((lambda_0 * epsilon_0 + nu_sum[c_i, z_i]) / lamb)  # % 24
                    xi = xi_0 + lambda_0 * n / lamb * (nu_sum[c_i, z_i] / n - epsilon_0) ** 2 + (
                            sigma_sum[c_i, z_i] - nu_sum[c_i, z_i] ** 2 / n)
                    tau = tau_0 + n
                    sigma[c_i, z_i] = xi * (lamb + 1) / lamb / (tau - 1)
                else:
                    sigma[c_i, z_i] = invgamma.rvs(tau_0, xi_0)
                    nu[c_i, z_i] = np.random.normal(epsilon_0, np.sqrt(sigma[c_i, z_i]/lambda_0))
                    nu_sum[c_i, z_i] = 0
                    sigma_sum[c_i, z_i] = 0

            for r in range(region_num):
                if n_r[r] == 0:
                    p_l[r] = multivariate_normal.pdf(location, mu[r], Sigma[r])
                else:
                    p_l[r] = multivariate_t(location, mu[r], Sigma[r], n_r[r]+rho_0-1)

            p_r = (n_r + alpha)/sum(n_r + alpha)
            p = np.multiply(p_r, p_l)

            c_i = d['cat'+str(category_num)]
            for r in range(region_num):
                p[r] *= (n_rc[r, c_i] + beta) / sum(n_rc[r, :] + beta)
            p = p/np.sum(p)
            r_i = np.where(np.random.multinomial(1, p))[0]
            p_z = np.zeros(period_num, dtype=np.float64)
            for z in range(period_num):
                p_z[z] += (n_cz[c_i, z] + gamma) / sum(n_cz[c_i, :] + gamma)
                if n_cz[c_i, z] == 0:
                    p_z[z] *= norm.pdf(time, nu[c_i, z], sigma[c_i, z])
                else:
                    p_z[z] *= t.pdf((time-nu[c_i, z])/np.sqrt(sigma[c_i, z]), n_cz[c_i, z]+tau_0)
            p_z = (p_z + 1e-15) / np.sum(p_z + 1e-15)
            z_i = np.where(np.random.multinomial(1, p_z))[0]

            record[i]['c'] = c_i
            record[i]['z'] = z_i
            record[i]['r'] = r_i
            n_r[r_i] += 1
            n = n_r[r_i]
            mu_sum[r_i] += location
            mu_sum_mat = mu_sum[r_i].T
            Sigma_sum[r_i] += l_mat*np.transpose(l_mat)
            kappa = kappa_0 + n
            mu[r_i] = (kappa_0*upsilon_0 + mu_sum[r_i]) / kappa
            zeta = zeta_0 + kappa_0*n / kappa * (
                    (mu_sum_mat / n - upsilon_0_mat)*np.transpose(mu_sum_mat/n-upsilon_0_mat)) + (
                    Sigma_sum[r_i] - mu_sum_mat*np.transpose(mu_sum_mat)/n)
            if ~np.all(np.linalg.eigvals(zeta) > 0):
                zeta = zeta_0
            rho = rho_0 + n
            Sigma[r_i] = zeta * (kappa + 1) / kappa / (rho - 1)
            n_c[c_i] += 1
            n_rc[r_i, c_i] += 1
            n_urc[d['u'] - 1, r_i, c_i] += 1
            n_cz[c_i, z_i] += 1
            n = n_cz[c_i, z_i]
            nu_sum[c_i, z_i] += time
            sigma_sum[c_i, z_i] += time ** 2
            lamb = lambda_0 + n
            nu[c_i, z_i] = ((lambda_0*epsilon_0 + nu_sum[c_i, z_i]) / lamb)
            xi = xi_0 + lambda_0*n/lamb*(nu_sum[c_i, z_i]/n-epsilon_0)**2 + (
                    sigma_sum[c_i, z_i] - nu_sum[c_i, z_i]**2/n)
            tau = tau_0 + n
            sigma[c_i, z_i] = xi * (lamb + 1) / lamb / (tau - 1)

            i += 1

        time_end = timer.time()
        print(time_end-time_start)

    n_ct = np.zeros((category_num, 28), dtype=int)
    for d in record:
        longitude = d['longitude']
        latitude = d['latitude']
        location = np.array([longitude, latitude])
        p_r_max = 0
        for r in range(region_num):
            p_r = multivariate_t(location, mu[r], Sigma[r], n_r[r] + rho_0 - 1)
            if p_r > p_r_max:
                p_r_max = p_r
                r_i = r
        n_urc[d['u'] - 1, r_i, d['cat'+str(category_num)]] += 1
        n_rc[r_i, d['cat'+str(category_num)]] += 1
        n_ct[d['cat'+str(category_num)], int(d['time'])] += 1
    np.save('parameters/n_rc.npy', n_rc)
    np.save('parameters/n_urc.npy', n_urc)
    np.save('parameters/n_ct.npy', n_ct)
    np.save('parameters/mu.npy', mu)
    np.save('parameters/SSigma.npy', Sigma)
    np.save('parameters/n_cz.npy', n_cz)
    np.save('parameters/nu.npy', nu)
    np.save('parameters/sigma.npy', sigma)

    Phi_r = (n_rc+beta) / np.repeat((n_rc+beta).sum(axis=1), category_num).reshape(region_num, category_num).astype(float)
    for user_iter in range(user_num):
        print('User %d' % user_iter)
        time_start = timer.time()
        Phi_ur = Phi_r * b + n_urc[user_iter, :, :]
        n_r_u = n_r
        n_cz_u = n_cz
        n_c_u = n_c
        mu_u = mu
        Sigma_u = Sigma
        nu_u = nu
        sigma_u = sigma
        mu_sum_u = mu_sum
        Sigma_sum_u = Sigma_sum
        nu_sum_u = nu_sum
        sigma_sum_u = sigma_sum
        data = np.load(city + '_dataset/userdata/usertrace_'+str(user_iter+1)+'.npy')
        dataset = np.array([], dtype=Record_sample)
        for d in data:
            dataset = np.concatenate([dataset,
                                  np.array([(d['longitude'], d['latitude'], d['time'], d['weekend'], d['POI'], -1, -1, -1, -1, -1, -1, d['user'])],
                                           dtype=Record_sample)])
        for gibbs_iter in range(gibbs_iter_num):
            p_l = np.zeros(region_num, dtype=np.float64)
            i = 0
            for d in dataset:
                time = d['time']
                location = np.array([d['longitude'], d['latitude']])
                l_mat = location[:, np.newaxis]
                if gibbs_iter > 0:
                    r_i = d['r']
                    c_i = d['c']
                    z_i = d['z']
                    n_r_u[r_i] -= 1
                    Phi_ur[r_i, c_i] -= 1
                    n_c_u[c_i] -= 1
                    n_cz_u[c_i, z_i] -= 1

                    if n_r_u[r_i] > 0:
                        n = n_r_u[r_i]
                        mu_sum_u[r_i] -= location
                        mu_sum_mat = mu_sum_u[r_i]
                        mu_sum_mat = mu_sum_mat[:, np.newaxis]
                        Sigma_sum_u[r_i] -= l_mat*np.transpose(l_mat)
                        kappa = kappa_0 + n
                        mu_u[r_i] = (kappa_0*upsilon_0 + mu_sum_u[r_i]) / kappa
                        zeta = zeta_0 + kappa_0*n / kappa * (
                                (mu_sum_mat / n - upsilon_0_mat)*np.transpose(mu_sum_mat/n-upsilon_0_mat)) + (
                                Sigma_sum_u[r_i] - mu_sum_mat*np.transpose(mu_sum_mat)/n)
                        if ~np.all(np.linalg.eigvals(zeta) > 0):
                            zeta = zeta_0
                        rho = rho_0 + n
                        Sigma_u[r_i] = zeta * (kappa + 1) / kappa / (rho - 1)
                    else:
                        Sigma_u[r_i] = invwishart.rvs(rho_0, zeta_0)
                        mu_u[r_i] = np.random.multivariate_normal(upsilon_0, Sigma_u[r_i]/kappa_0)
                        mu_sum_u[r_i] = 0
                        Sigma_sum_u[r_i] = np.zeros((2, 2), dtype=np.float64)

                    if n_cz_u[c_i, z_i] > 0:
                        n = n_cz_u[c_i, z_i]
                        nu_sum_u[c_i, z_i] -= time
                        sigma_sum_u[c_i, z_i] -= time ** 2
                        lamb = lambda_0 + n
                        nu_u[c_i, z_i] = ((lambda_0 * epsilon_0 + nu_sum_u[c_i, z_i]) / lamb)  # % 24
                        xi = xi_0 + lambda_0 * n / lamb * (nu_sum_u[c_i, z_i] / n - epsilon_0) ** 2 + (
                                sigma_sum_u[c_i, z_i] - nu_sum_u[c_i, z_i] ** 2 / n)
                        tau = tau_0 + n
                        sigma_u[c_i, z_i] = xi * (lamb + 1) / lamb / (tau - 1)
                    else:
                        sigma_u[c_i, z_i] = invgamma.rvs(tau_0, xi_0)
                        nu_u[c_i, z_i] = np.random.normal(epsilon_0, np.sqrt(sigma_u[c_i, z_i]/lambda_0))
                        nu_sum_u[c_i, z_i] = 0
                        sigma_sum_u[c_i, z_i] = 0

                for r in range(region_num):
                    if n_r_u[r] == 0:
                        p_l[r] = multivariate_normal.pdf(location, mu_u[r], Sigma_u[r])
                    else:
                        p_l[r] = multivariate_t(location, mu_u[r], Sigma_u[r], n_r_u[r]+rho_0-1)

                p_r = np.multiply((n_r_u + alpha)/sum(n_r_u + alpha), p_l)
                p_rcz = np.tile(p_r[:, np.newaxis], (1, category_num*period_num))

                p_c = np.tile(Phi_ur, (1, period_num))
                p_rcz = np.multiply(p_rcz, p_c)

                p_cz = np.zeros((period_num, category_num), dtype=np.float64)
                for c in range(category_num):
                    for z in range(period_num):
                        p_cz[z, c] *= (n_cz_u[c, z] + gamma) / sum(n_cz_u[c, :] + gamma)
                        if n_cz_u[c, z] == 0:
                            p_cz[z, c] *= norm.pdf(time, nu_u[c, z], np.sqrt(sigma_u[c, z]))
                        else:
                            p_cz[z, c] *= t.pdf((time-nu_u[c, z])/np.sqrt(sigma_u[c, z]), n_cz_u[c, z]+tau_0)
                p_cz = np.reshape(p_cz, (1, -1))
                p_cz = np.tile(p_cz, (region_num, 1))
                p_rcz = np.multiply(p_rcz, p_cz)

                p = np.reshape(p_rcz, (1, -1))[0]
                p = (p + 1e-15) / np.sum(p + 1e-15)
                rcz_i = np.where(np.random.multinomial(1, p))[0]
                r_i = rcz_i // (category_num * period_num)
                cz_i = rcz_i % (category_num * period_num)
                c_i = cz_i % category_num
                z_i = cz_i // category_num

                dataset[i]['c'] = c_i
                dataset[i]['z'] = z_i
                dataset[i]['r'] = r_i
                n_r_u[r_i] += 1
                Phi_ur[r_i, c_i] += 1
                n = n_r_u[r_i]
                mu_sum_u[r_i] += location
                mu_sum_mat_u = mu_sum_u[r_i].T
                Sigma_sum_u[r_i] += l_mat*np.transpose(l_mat)
                kappa = kappa_0 + n
                mu_u[r_i] = (kappa_0*upsilon_0 + mu_sum_u[r_i]) / kappa
                zeta = zeta_0 + kappa_0*n / kappa * (
                        (mu_sum_mat_u / n - upsilon_0_mat)*np.transpose(mu_sum_mat_u/n-upsilon_0_mat)) + (
                        Sigma_sum_u[r_i] - mu_sum_mat_u*np.transpose(mu_sum_mat_u)/n)
                if ~np.all(np.linalg.eigvals(zeta) > 0):
                    zeta = zeta_0
                rho = rho_0 + n
                Sigma_u[r_i] = zeta * (kappa + 1) / kappa / (rho - 1)

                n_c_u[c_i] += 1
                n_cz_u[c_i, z_i] += 1
                n = n_cz_u[c_i, z_i]
                nu_sum_u[c_i, z_i] += time
                sigma_sum_u[c_i, z_i] += time ** 2
                lamb = lambda_0 + n
                nu_u[c_i, z_i] = ((lambda_0*epsilon_0 + nu_sum_u[c_i, z_i]) / lamb)
                xi = xi_0 + lambda_0*n/lamb*(nu_sum_u[c_i, z_i]/n-epsilon_0)**2 + (
                        sigma_sum_u[c_i, z_i] - nu_sum_u[c_i, z_i]**2/n)
                tau = tau_0 + n
                sigma_u[c_i, z_i] = xi * (lamb + 1) / lamb / (tau - 1)

                i += 1
        time_end = timer.time()
        print(time_end-time_start)
        np.save('parameters/Phi_ur' + str(user_iter) + '.npy', Phi_ur)


if __name__ == '__main__':
    main()
