import numpy as np
import math
import matplotlib.pyplot as plt

# Sample points in a disk
def sampleFromDisk(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.rand(2*n,2)*2*r-r
    
    array = np.multiply(array.T,(np.linalg.norm(array,2,axis=1)<r)).T
    array = array[~np.all(array==0, axis=1)]
    
    if np.shape(array)[0]>=n:
        return array[0:n]
    else:
        return sampleFromDisk(r,n)

def sampleFromDomain(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    array = np.zeros([n,2])
    c = np.array([0.3,0.0])
    r = 0.3

    for i in range(n):
        array[i] = randomPoint(c,r)

    return array

def randomPoint(c,r):
    point = np.random.rand(2)*2-1
    if np.linalg.norm(point-c)<r:
        return randomPoint(c,r)
    else:
        return point

def sampleFromBoundary(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    c = np.array([0.3,0.0])
    r = 0.3
    length = 4*2+2*math.pi*r
    interval1 = np.array([0.0,2.0/length])
    interval2 = np.array([2.0/length,4.0/length])
    interval3 = np.array([4.0/length,6.0/length])
    interval4 = np.array([6.0/length,8.0/length])
    interval5 = np.array([8.0/length,1.0])

    array = np.zeros([n,2])

    for i in range(n):
        rand0 = np.random.rand()
        rand1 = np.random.rand()

        point1 = np.array([rand1*2.0-1.0,-1.0])
        point2 = np.array([rand1*2.0-1.0,+1.0])
        point3 = np.array([-1.0,rand1*2.0-1.0])
        point4 = np.array([+1.0,rand1*2.0-1.0])
        point5 = np.array([c[0]+r*math.cos(2*math.pi*rand1),c[1]+r*math.sin(2*math.pi*rand1)])

        array[i] = myFun(rand0,interval1)*point1 + myFun(rand0,interval2)*point2 + \
            myFun(rand0,interval3)*point3 + myFun(rand0,interval4)*point4 + \
                myFun(rand0,interval5)*point5
 
    return array

def myFun(x,interval):
    if interval[0] <= x <= interval[1]:
        return 1.0
    else: return 0.0

def sampleFromSurface(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,2))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromSurface(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        return array*r

# Sample from 10d-ball
def sampleFromDisk10(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,10))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromDisk10(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        radius = np.random.rand(n,1)**(1/10)
        array = np.multiply(array,radius)

        return r*array

def sampleFromSurface10(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,10))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromSurface10(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        return array*r



def sampleFromSquare(a, n):
    """
    Samples n points uniformly from a square [-a, a] x [-a, a].
    
    Parameters:
    a -- Half the length of the square's side.
    n -- Number of samples.
    
    Returns:
    A numpy array of shape (n, 2) containing sampled points.
    """
    return np.random.uniform(low=0, high=a, size=(n, 2))

def sampleFromSquareBoundary(a, n):
    """
    Samples n points uniformly from the boundary of a square [-a, a] x [-a, a].
    
    Parameters:
    a -- Half the length of the square's side.
    n -- Number of samples.
    
    Returns:
    A numpy array of shape (n, 2) containing sampled boundary points.
    """
    edges = np.random.choice(4, n)  # Choose an edge for each sample
    positions = np.random.uniform(0, a, n)  # Position along the edge
    samples = np.zeros((n, 2))
    
    mask0 = edges == 0
    mask1 = edges == 1
    mask2 = edges == 2
    mask3 = edges == 3
    
    samples[mask0] = np.column_stack((positions[mask0], np.full(np.sum(mask0), a))) # Top edge
    samples[mask1] = np.column_stack((positions[mask1], np.full(np.sum(mask1), 0))) # Bottom edge
    samples[mask2] = np.column_stack((np.full(np.sum(mask2), a), positions[mask2]))  # Right edge
    samples[mask3] = np.column_stack((np.full(np.sum(mask3), 0), positions[mask3])) # Left edge
    
    return samples

import numpy as np

def sampleFromLShape(a, n):
    """
    Samples n points uniformly from an L-shaped region in [0, a] x [0, a]
    with the upper-right quadrant [a/2, a] x [a/2, a] removed.

    Parameters:
    a -- Side length of the square.
    n -- Number of samples.

    Returns:
    A numpy array of shape (n, 2) containing sampled points.
    """
    samples = []
    while len(samples) < n:
        x, y = np.random.uniform(0, a, 2)
        if not (x >= a / 2 and y >= a / 2):  # Exclude the upper-right quadrant
            samples.append((x, y))
    return np.array(samples)


def sampleFromLShapeBoundary(a, n):
    """
    Samples n points uniformly from the boundary of an L-shaped region in [0, a] x [0, a]
    with the upper-right quadrant [a/2, a] x [a/2, a] removed.

    Parameters:
    a -- Side length of the square.
    n -- Number of samples.

    Returns:
    A numpy array of shape (n, 2) containing sampled boundary points.
    """
    edges = [
        (0, 0, a, 0),  # Bottom edge
        (0, 0, 0, a),  # Left edge
        (0, a, a/2, a),  # Top-left edge
        (a, a/2, a, 0),  # Right edge
        (a/2, a/2, a/2, a),  # Inner cut vertical
        (a/2, a/2, a, a/2)  # Inner cut horizontal
    ]
    
    samples = []
    edge_lengths = [np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in edges]
    total_length = sum(edge_lengths)
    probs = np.array(edge_lengths) / total_length  # Probability proportional to length
    
    while len(samples) < n:
        edge_idx = np.random.choice(len(edges), p=probs)  # Choose edge based on probability
        x1, y1, x2, y2 = edges[edge_idx]
        t = np.random.uniform(0, 1)
        x, y = (1 - t) * np.array([x1, y1]) + t * np.array([x2, y2])  # Linear interpolation
        samples.append((x, y))
    
    return np.array(samples)


if __name__ == "__main__":
    array = sampleFromLShapeBoundary(1,1000).T
    # array = sampleFromBoundary(500).T
    plt.plot(array[0],array[1],'o',ls="None")
    plt.axis("equal")
    plt.show()
    pass