import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa



def generate_data():
    np.random.seed(421)
    X1 = np.random.multivariate_normal(np.array([+2.5, +2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]), 50)
    X2 = np.random.multivariate_normal(np.array([-2.5, +2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]), 50)
    X3 = np.random.multivariate_normal(np.array([-2.5, -2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]), 50)
    X4 = np.random.multivariate_normal(np.array([2.5, -2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]), 50)
    X5 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1.6, 0], [0, 1.6]]), 100)
    X = np.vstack((X1, X2, X3, X4, X5))
    return X

def plot_generated_data(X):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(X[:, 0], X[:, 1], "b.", markersize=10)

    plt.xlabel("x1")
    plt.ylabel("x2")

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K),:]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


def calculate_prob(mean, covariance, point):
    return np.sqrt(1/np.power(2*np.pi,2)/np.linalg.det(covariance))*np.exp(-1/2*np.dot(np.dot(np.transpose(point-mean), np.linalg.inv(covariance)), (point-mean)))



def plot_current_state(centroids, covariance, memberships ,X):

    x1_interval = np.linspace(-6, +6, 60)
    x2_interval = np.linspace(-6, +6, 60)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

    discriminant_values = np.zeros((len(x1_interval), len(x2_interval), 10))

    for i, x2 in enumerate(x2_interval):
        for j, x1 in enumerate(x1_interval):
            for c in range(5):
                discriminant_values[i,j,c] = calculate_prob(centroids[c],covariance[c], [x1,x2])

    for i, x2 in enumerate(x2_interval):
        for j, x1 in enumerate(x1_interval):
            discriminant_values[i ,j ,5] = calculate_prob(np.array([+2.5, +2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]),
                                                          [x1, x2])
            discriminant_values[i, j, 6] = calculate_prob(np.array([-2.5, +2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]),
                                                          [x1, x2])
            discriminant_values[i, j, 7] = calculate_prob(np.array([-2.5, -2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]),
                                                          [x1, x2])
            discriminant_values[i, j, 8] = calculate_prob(np.array([2.5, -2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]),
                                                          [x1, x2])
            discriminant_values[i, j, 9] = calculate_prob(np.array([0, 0]), np.array([[1.6, 0], [0, 1.6]]),
                                                          [x1, x2])
    for c in range(5):
        plt.contour(x1_grid, x2_grid, discriminant_values[:, :, c] - 0.05, levels=0, colors="k")

    for c in range(5):
        plt.contour(x1_grid, x2_grid, discriminant_values[:, :, c+5] - 0.05, levels=0, colors="k", linestyles="dashed")

    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])



    plt.xlabel("x1")
    plt.ylabel("x2")


def iterate_K(X):
    centroids = None
    memberships = None
    iteration = 1
    while iteration<3:


        old_centroids = centroids
        centroids = update_centroids(memberships, X)
        if np.alltrue(centroids == old_centroids):
            break


        old_memberships = memberships
        memberships = update_memberships(centroids, X)
        if np.alltrue(memberships == old_memberships):
            break



        iteration = iteration + 1
    return centroids, memberships

def one_hot_encode(X,memberships):
    coded = np.asarray([[int(memberships[n]==c) for c in range(K)] for n in range(X.shape[0])])
    return coded

def calculate_covariance(X,means,memberships, class_sizes):
    covariances = np.array([(np.dot(np.transpose(X[memberships == c, :] - (means[c, :])),
                                    X[memberships == c, :] - (means[c, :]))) / class_sizes[c]
                            for c in range(K)])
    return covariances

def calculate_prior( memberships):
    class_priors = [np.mean(memberships == c) for c in range(K)]
    return class_priors

def find_class_sizes(coded):
    class_sizes = np.sum(coded, axis=0)
    return class_sizes

def iterate_em(prior, mean, covariance, X, K):
    prior = prior
    mean = mean
    covariance = covariance
    iteration = 1
    while iteration<101:
        e_probs = np.asarray([[np.exp(-1/2*np.log(np.linalg.det(covariance[c]))-1/2*np.dot(np.dot((X[n]-mean[c]).transpose(),np.linalg.inv(covariance[c])), (X[n]-mean[c])) + np.log(prior[c]))
                               for c in range(K)] for n in range(X.shape[0])]).reshape(300,5)
        e_probs = e_probs/ (np.sum(e_probs, axis=1)).reshape(300,1)

        mean = np.asarray([np.sum(np.multiply(e_probs[:,c].reshape(300,1) , X), 0)/np.sum(e_probs[:,c].reshape(300,1)) for c in range(K)])

        covariance = np.asarray([ np.dot(np.multiply(e_probs[:,c].transpose(), np.transpose(X - mean[c,:])), (X - mean[c,:]))
                                  /np.sum(e_probs[:,c], 0) for c in range(K)])

        prior = np.asarray([np.sum(e_probs[:,c], 0)/X.shape[0] for c in range(K)])

        iteration = iteration + 1

    memberships = find_membership(e_probs)
    plt.subplot(1, 2, 2)
    plot_current_state(mean, covariance, memberships, X)

    coded = one_hot_encode(X, memberships)
    print(find_class_sizes(coded))
    print("############")
    print(mean)
    print("############")
    print(covariance)

def find_membership(probs):
    return np.argmax(probs, axis=1)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    X = generate_data()
    plot_generated_data(X)
    N = X.shape[0]
    K = 5
    centroids, memberships = iterate_K(X)
    coded = one_hot_encode(X, memberships)
    class_sizes = find_class_sizes(coded)
    covariance = calculate_covariance(X, centroids, memberships, class_sizes)
    priors = calculate_prior(memberships)
    iterate_em(priors, centroids, covariance, X, K)
    plt.show()




