import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm

width, height = 640, 480

N = 100             # number of birds
minDist = 15.0      # min dist of approach
maxRuleVel = 0.5    # max magnitude of velocities calculated by "rules"
maxVel = 7.0        # max magnitude of final velocity


class Birds:
    """
    Simulates flock behaviour of birds, using the realistic-looking Boids model (1986)
    """
    def __init__(self):
        self.N = N
        self.minDist = minDist
        self.maxRuleVel = maxRuleVel
        self.maxVel = maxVel

        # Computing initial position and velocity
        self.pos = [width / 2.0, height / 2.0] + 10 * np.random.rand(2 * N).reshape(N, 2)
        # Create an array of N random variable angles in the range [0. 2pi]
        angles = 2 * math.pi * np.random.rand(N)
        # Random velocity vector [x,y] coordinates zip grouped
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))

    def tick(self, frameNum, pts, beak):
        """
        Update the simulation by one time step
        """
        # get pairwise distances
        self.distMatrix = squareform(pdist(self.pos))
        # apply rules:
        self.vel += self.apply_rules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.apply_bc()
        # update data
        pts.set_data(self.pos.reshape(2 * self.N)[::2],
                     self.pos.reshape(2 * self.N)[1::2])
        vec = self.pos + 10 * self.vel / self.maxVel
        beak.set_data(vec.reshape(2 * self.N)[::2],
                      vec.reshape(2 * self.N)[1::2])

    def limit_vec(self, vec, max_val):
        """ Limit magnitude of 2D vector """
        mag = norm(vec)
        if mag > max_val:
            vec[0], vec[1] = vec[0] * max_val / mag, vec[1] * max_val / mag

    def limit(self, x, max_val):
        """ Limit magnitide of 2D vectors in array X to maxValue """
        for vec in x:
            self.limit_vec(vec, max_val)

    def apply_bc(self):
        """ Apply boundary conditions """
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR

    def apply_rules(self):
        # apply rule #1 - Separation
        D = self.distMatrix < 25.0
        vel = self.pos * D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        # different distance threshold
        D = self.distMatrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2

        # apply rule #1 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel


def tick(frameNum, pts, beak, birds):
    """ Update function for animation """
    birds.tick(frameNum, pts, beak)
    return pts, beak


def main():
    print('Starting flock simulation...')

    # Create birds
    birds = Birds()

    # Setup plot
    fig = plt.figure()
    ax = plt.axes(xlim=(0, width), ylim=(0, height))
    pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4, c='r', marker='o', ls='None')
    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, birds), interval=20)

    # TODO: add a "button press" event handler to scatter birds

    plt.show(anim)


if __name__ == '__main__':
    main()