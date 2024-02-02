import numpy as np
from functools import partial
from scipy.stats import norm
from scipy.special import softmax
from tensorflow.keras.losses import BinaryCrossentropy
from matplotlib import pyplot as plt

class SubPopulation:

    def __init__(self, mu, sigmas, fn):
        self.mus = [mu]
        self.sigmas = sigmas
        self.fn = fn

    def apply(self, n, theta):
        samples = self.sample(n)
        out = samples @ theta
        mu, self.sigmas = self.fn(out, self.mus[-1], self.sigmas)
        self.mus.append(mu)

    def sample(self, n):
        samples = []
        for mu, sigma in zip(self.mus[-1], self.sigmas):
            samples.append(np.random.normal(mu, sigma, n))
        return np.stack(samples).reshape(-1, self.mus[-1].shape[0])
    
    def plot_mus(self):
        plt.plot(self.mus)
        plt.title('Actual Means')
        plt.xlabel('Timestep')
        plt.ylabel('Mean')
        plt.show()

class MeanWorld:
    # the mean of each subpop evolves, new individuals are sampled
    def __init__(self, populations, n, weights):
        self.pops = populations
        self.n = n
        self.mus = []
        self.weights = weights

    def step(self, theta):
        for pop in self.pops:
            pop.apply(self.n, theta)
        self.estimate_mu()

    def sample(self):
        samples = []
        for pop, w in zip(self.pops, self.weights):
            samples.append(pop.sample(int(self.n * w)))
        samples = np.concatenate(samples)
        return samples
    
    def estimate_mu(self):
        samples = self.sample()
        self.mus.append(np.mean(samples, axis=0))

class StatefulPerfGD:

    def __init__(self, world, theta_init, H, theta_star):
        self.world = world
        self.thetas = [theta_init]
        self.mu_hats = []
        self.H = H
        self.theta_star = theta_star # the 'real' theta
        self.loss = BinaryCrossentropy(from_logits=False)

    def estimate_mean(self, samples):
        mu_hat = np.mean(samples, axis=0)
        self.mu_hats.append(mu_hat)

    def deploy_sample(self):
        self.world.step(self.thetas[-1]) # deploy theta and population reacts
        samples = self.world.sample() # collect samples
        self.estimate_mean(samples) # estimate mean
        return samples

    def estimate_partials(self, psi, mu_hat, t):
        psis = []
        mu_hats = []
        for t in range(self.H, 0, -1):
            psis.append(np.concatenate((self.world.mus[-t], self.thetas[-t])))
            mu_hats.append(self.mu_hats[t])
        psi_H = np.stack(psis)
        mu_hats = np.stack(mu_hats)
        dpsi = psi_H - psi         # subtract psi from each prev timestep psi
        dmu_hat = mu_hats - mu_hat
        adj = np.conj(dpsi).T @ dmu_hat
        # extract estimates of deriv of m wrt first and second inputs (theta, mu)
        half = adj.shape[0] // 2
        d1, d2 = adj[:half], adj[half:]
        return d1, d2

    def estimateLTJac(self, d1, d2):
        # estimate derivative of long term mu w.r.t theta
        term1 = np.eye(d2.shape[0]) - d2
        dmu_star = np.linalg.pinv(term1) @ d1
        return dmu_star

    def estimateLTGrad(self, samples, theta, mu_hat, dmu_star):
        # for each input z
            # gradient of loss wrt theta * pdf(z, mu_hat) + loss * du/dtheta * grad of pdf wrt mu
        
        # getting intermediate values
        y_true = np.clip(samples @ self.theta_star, 0, 1)
        y_pred = softmax(samples @ theta)
        gloss = samples.T @ (y_pred - y_true)
        pdfs = []
        grad_add = (mu_hat - samples)
        for dim in range(samples.shape[1]):
            pdf = norm.pdf(samples[:,dim], loc=mu_hat[dim], scale=1.0)
            pdfs.append(pdf)
        pdfs = np.stack(pdfs).T
        gpdfs = pdfs * grad_add
        loss = self.loss(y_true, y_pred)

        ltgrad = np.sum(gloss[:,np.newaxis] * pdfs.T + loss * dmu_star @ gpdfs.T, axis=1)
        return ltgrad, loss

    def spgd(self, lr, max_steps):

        # collect for plotting
        losses = []

        # start SPGD algorithm
        samples = self.world.sample()
        self.estimate_mean(samples)
        theta = theta_init
        for t in range(max_steps):
            psi = np.concatenate((self.mu_hats[-1], self.thetas[-1]))
            samples = self.deploy_sample()
            mu_hat = self.mu_hats[-1]

            if t >= self.H:
                d1, d2 = self.estimate_partials(psi, mu_hat, t)
                dmu_star = self.estimateLTJac(d1, d2)
                grad, loss = self.estimateLTGrad(samples, theta, mu_hat, dmu_star)
                losses.append(loss)

                # estimation noise: optional
                noise = 0   # np.random.normal(loc=0, scale=1, size=grad.shape)
                theta = self.thetas[-1] - lr * (grad + noise)
            self.thetas.append(theta)
        return losses
    
    def plot_mus(self):
        plt.plot(self.mu_hats)
        plt.xlabel('Timestep')
        plt.ylabel('Predicted Mean')
        plt.title('Simulation Predicted Means')
        plt.show()


# example setup with 2 simple populations
# identical starting states
# pop2 has higher barrier of entry
def fn1(k, samples, mus, sigmas):
    acceptance_rate = np.mean(samples > k)
    mus = mus + (mus * acceptance_rate * .1)
    return mus, sigmas

arr = np.array([1, 1, 1])
add = np.clip(np.random.normal(), 0, None)
print(arr + add)
pop1 = SubPopulation(arr + add, arr, partial(fn1, 0))
pop2 = SubPopulation(arr, arr, partial(fn1, 0))
world = MeanWorld([pop1, pop2], 100, [.5, .5])

# init simulation, Horizon = 1
theta_init = np.random.normal(size=(3,))
theta_star = np.random.normal(size=(3,))
simulation = StatefulPerfGD(world, theta_init, 6, theta_star)
losses = simulation.spgd(.001, 100)

# plot
plt.plot(losses)
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss')
plt.show()
simulation.plot_mus()

pop1.plot_mus()
pop2.plot_mus()