import numpy as np
import matplotlib.pyplot as plt

coin_flips=1001
sequences = 100000
p = 0.25

data = np.random.binomial(1, 0.25, (sequences, coin_flips))
epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]

# Article a
mean = np.ndarray(shape=(5, coin_flips))
for m in range(1, coin_flips):
    mean[..., m] = (data[:5, :m]).mean(axis=1)

plt.plot(mean[0, 1:])
plt.plot(mean[1, 1:])
plt.plot(mean[2, 1:])
plt.plot(mean[3, 1:])
plt.plot(mean[4, 1:])
plt.title('5 Coin toss (p=0.25) Sequences Mean VS m', fontsize=18)
plt.xlabel('m - number of tosses the mean is taken on', fontsize=14)
plt.ylabel('Mean', fontsize=14)
plt.savefig('a')
plt.show()

# Article b
for epsilon in epsilons:
    m = np.arange(start=1,stop=coin_flips,step=1)
    chebychev_upper_bound = np.minimum(1 / (4 * m * (epsilon**2)), 1)
    hoeffding_upper_bound = np.minimum(2 * np.exp(-2 * m * (epsilon**2)), 1)

    fig = plt.figure()

    plt.title('Probability Upper Bounds - epsilon=' + str(epsilon), fontsize=18)
    plt.xlabel('m - number of tosses Mean is taken on', fontsize=14)
    plt.ylabel('Probability Upper Bound', fontsize=14)

    plt.plot(m, chebychev_upper_bound, label='Chebychev''s Upper Bound', alpha=0.5, linewidth=3.0)
    plt.plot(m, hoeffding_upper_bound, label='Hoeffding''s Upper Bound', alpha=0.5, linewidth=3.0)
    plt.savefig('article_b_epsilon='+str(epsilon).replace(".", "_"))
    plt.legend()
    plt.show()

    # Article c
#     dist = np.ndarray(shape=(sequences, coin_flips))
#     for i in range(1, coin_flips):
#         dist[..., i] = np.abs((data[..., :i]).mean(axis=1) - p)
#
#     satisfy = (np.count_nonzero(dist[..., 1:] >= epsilon, axis=0) / sequences) * 100
#
#     plt.figure()
#     plt.plot(m, satisfy, alpha=0.5, linewidth=3.0)
#     plt.title('Percentages of Sequences out of bounds VS m - epsilon=' + str(epsilon), fontsize=16)
#     plt.xlabel('m - number of tosses Mean is taken on', fontsize=14)
#     plt.ylabel('Percentage %')
# #    plt.savefig('Percentage='+str(epsilon).replace(".", "_"), figsize=(10,10))
#
#     plt.show()


