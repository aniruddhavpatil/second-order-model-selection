#!/usr/local/bin/python3

import os
import numpy as np
from matplotlib import pyplot as plt
import random
import time

# Reads the data present in the folder (.csv files only)
def read_data():
    # curr_dir = os.getcwd()
    data_dir = os.path.join(os.getcwd(), 'data')
    dir_files = os.listdir(data_dir)
    data = {}
    for file in dir_files:
        if file.endswith('.csv'):
            with open(os.path.join(data_dir,file), 'r') as f:
                file_key = file.split('.')[0]
                data[file_key] = np.genfromtxt(f, delimiter=',', dtype='float32')

    return data

# Regularization as per the equation (3.28) in Bishop's text
def regularization(phi, t, lamb):
    phiT_phi = phi.T @ phi
    I = np.identity(phiT_phi.shape[0])
    w = np.linalg.inv(lamb * I + phiT_phi) @ phi.T @ t
    return w

# MSE evaluation as per instructions
def MSE(w, phi, t):
    return (np.linalg.norm(phi @ w - t) ** 2) / len(t)


# Plot for Task 1
def plot(variant, y):
    plt.clf()
    x = np.arange(0, len(y))
    plt.xlabel('Regularization parameter (lambda) on the ' + variant + ' dataset')
    plt.ylabel('MSE')
    x_ticks = [x for x in range(0, 151, 10)]
    plt.xticks(x_ticks)
    for i in x_ticks:
        plt.axvline(i, color='grey', linestyle='-', linewidth=0.5)
    plt.plot(x, [i[0] for i in y], label='train')
    plt.plot(x, [i[1] for i in y], label='test')
    plt.legend(loc='best')
    plt.savefig(variant + '.png')

# Shuffle dataset along with labels (based on common indices)
def dataset_shuffle(phi, t):
    indices = [i for i in range(len(t))]
    random.shuffle(indices)
    shuffled_phi = np.empty((phi.shape))
    shuffled_t = np.empty((t.shape[0],))
    count = 0
    for i in indices:
        shuffled_phi[count] = phi[i]
        shuffled_t[count] = t[i]
        count += 1

    return shuffled_phi, shuffled_t

# Conduct Task 1
def regularization_task(data, variants):
    for variant in variants:
        train_matrix = data['train-' + variant]
        train_labels = data['trainR-' + variant]

        test_matrix = data['test-' + variant]
        test_labels = data['testR-' + variant]
        mse_vals = []
        for i in range(151):
            w = regularization(train_matrix, train_labels, i)
            mse_train = MSE(w, train_matrix, train_labels)
            mse_test = MSE(w, test_matrix, test_labels)
            mse_vals.append((mse_train, mse_test))

        plot(variant, mse_vals)

# Plot for Task 2
def lc_plot(lamb, mse):
    plt.clf()
    plt.xlabel('Learning curve with lambda=' + str(lamb))
    plt.ylabel('MSE')
    x = [x for x in range(10, 1001, 10)]
    plt.plot(x, mse)
    plt.savefig('lc_lambda_' + str(lamb) + '.png')

# Conducts task 2
def generate_learning_curves(data, variant):
    small = 1
    right = 20
    large = 150
    lamb_mse = {
        small: [],
        right: [],
        large: []
    }

    train_matrix = data['train-' + variant]
    train_labels = data['trainR-' + variant]
    test_matrix = data['test-' + variant]
    test_labels = data['testR-' + variant]

    for i in range(10):

        shuffled_train_matrix, shuffled_train_labels = dataset_shuffle(train_matrix, train_labels)

        for lamb in [small, right, large]:
            mse_across_sizes = []
            for size in range(10, 1001, 10):
                w = regularization(shuffled_train_matrix[:size], shuffled_train_labels[:size], lamb)
                mse_across_sizes.append(MSE(w, test_matrix, test_labels))
            lamb_mse[lamb].append(mse_across_sizes)

    for i in lamb_mse:
        lamb_mse[i] = np.mean(lamb_mse[i], axis=0)
        lc_plot(i, lamb_mse[i])

# Conducts Task 3.1
def cross_validation_selection(data, variants, folds=10):
    for variant in variants:
        start = time.time()
        train_matrix = data['train-' + variant]
        train_labels = data['trainR-' + variant]
        test_matrix = data['test-' + variant]
        test_labels = data['testR-' + variant]

        per_fold_samples = train_matrix.shape[0] // folds

        mse_expt_vals = []
        for j in range(10):

            # Get shuffled data for every experiment
            shuffled_train_matrix, shuffled_train_labels = dataset_shuffle(train_matrix, train_labels)
            for i in range(folds):
                # Create train and test folds
                test_idx = (i * per_fold_samples, (i + 1) * per_fold_samples)
                phi_test = shuffled_train_matrix[test_idx[0]: test_idx[1]]
                t_test = shuffled_train_labels[test_idx[0]:test_idx[1]]

                phi_train = np.vstack(
                    (shuffled_train_matrix[: test_idx[0]],
                     shuffled_train_matrix[test_idx[1]:])
                )

                t_train = np.concatenate(
                    (shuffled_train_labels[: test_idx[0]],
                     shuffled_train_labels[test_idx[1]:])
                )

                mse_lamb_vals = []
                # Regularization for all lambdas
                for i in range(151):
                    w = regularization(phi_train, t_train, i)
                    mse = MSE(w, phi_test, t_test)
                    mse_lamb_vals.append(mse)
                mse_expt_vals.append(mse_lamb_vals)

        mse_avg = np.mean(mse_expt_vals, axis=0)
        # Retrieve selected lambda as the argmin of mse_avg
        selected_lambda = np.argmin(mse_avg)
        corresponding_mse = mse_avg[selected_lambda]
        # Train on whole train set
        w = regularization(train_matrix, train_labels, selected_lambda)
        mse = MSE(w, test_matrix, test_labels)
        print('Dataset:', variant, 'took:', time.time() - start, 'seconds')
        print('selected_lambda:', selected_lambda, 'train_mse:', corresponding_mse, 'test_mse', mse)

# Bayesian iteration for Task 3.2
def bayesian_iteration(phi, t, epsilon=1e-4):
    alpha = random.randrange(1, 10)
    beta = random.randrange(1, 10)
    phiT_phi = phi.T @ phi
    eig_val, _ = np.linalg.eig(phiT_phi)
    while True:
        sn = np.linalg.inv(alpha * np.identity(phiT_phi.shape[0]) + beta * phiT_phi)
        mn = beta * sn @ phi.T @ t
        gamma = 0
        for l in eig_val:
            gamma += l / (alpha + l)

        new_alpha = gamma / (mn.T @ mn)
        new_beta = (t.shape[0] - gamma) / np.linalg.norm(phi @ mn - t) ** 2
        total_difference = abs(new_alpha - alpha) + abs(new_beta - beta)
        if total_difference < epsilon:
            return new_alpha, new_beta
        else:
            alpha = new_alpha
            beta = new_beta

# Conducts Task 3.2
def bayesian_selection(data, variants):
    for variant in variants:
        start = time.time()
        phi = data['train-' + variant]
        t = data['trainR-' + variant]
        alpha, beta = bayesian_iteration(phi, t, epsilon=1e-4)
        w = regularization(phi, t, alpha / beta)
        mse = MSE(w, data['test-' + variant], data['testR-' + variant])
        print('variant:', variant, 'took:', time.time() - start, 'seconds', 'mse:', mse, 'selected_lambda:',
              alpha / beta)


def main():
    print('Parsing data...')
    start = time.time()
    data = read_data()
    end = time.time()

    print('Data parsed and loaded onto memory in', end - start, 'seconds')
    variants = ['100-10', '100-100', '1000-100', 'crime', 'wine']

    print('Starting regularization...')
    start = time.time()
    regularization_task(data, variants)
    end = time.time()
    print('Regularization completed in', end - start, 'seconds')

    print('Generating learning curves...')
    start = time.time()
    generate_learning_curves(data, '1000-100')
    end = time.time()
    print('Generated learning curves in', end - start, 'seconds')

    start = time.time()
    print('Starting cross-validation task...')
    cross_validation_selection(data, variants)
    end = time.time()
    print('Cross-validation task completed in', end - start, 'seconds')

    start = time.time()
    print('Starting bayesian model selection...')
    bayesian_selection(data, variants)
    end = time.time()
    print('Bayesian model selection completed in', end - start, 'seconds')


if __name__ == '__main__':
    main()
