import numpy as np
import matplotlib.pyplot as plt

# def spiralArrayBuilder(noElements, diameter, noArms, spacing):
#     a = 0
#     r = diameter
#     x = []
#     y = []
#     for i in range(noArms):
#         a = a + i
#         x.append([r*(np.cos(t+a)+t*np.sin(t+a)) for t in np.linspace(0,2*np.pi, num = int(2*noElements/noArms))]
#         y = [r*(np.sin(t+a)-t*np.cos(t+a)) for t in np.linspace(0,2*np.pi, num = int(2*noElements/noArms))]
#     print(np.linspace(0,2*np.pi, num = int(2*noElements/noArms)))
#     print(a)
#     wait = input("Press Enter to continue...")        
#     return [x,y]

def spiral_points(arc, separation):
    """generate points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive 
    turnings
    - approximate arc length with circle arc at given distance
    - use a spiral equation r = b * phi
    """
    def p2c(r, phi):
        """polar to cartesian
        """
        return (r * np.cos(phi), r * np.sin(phi))

    # yield a point at origin
    yield (0, 0)

    # initialize the next point in the required distance
    r = arc
    b = separation / (2 * np.pi)
    # find the first phi to satisfy distance of `arc` to the second point
    phi = float(r) / b
    while True:
        yield p2c(r, phi)
        # advance the variables
        # calculate phi that will give desired arc length at current radius
        # (approximating with circle)
        phi += float(arc) / r
        r = b * phi

# array = spiralArrayBuilder(64,8, 4, 0.1)
array = spiral_points(1, 0.1)
print(array)
plt.plot(array[0], array[1], 'o')
plt.show()
# length = len(array[0]) + len(array[1])
# print("Length of array:", length)
# noElements = 8
# elementPos =  np.zeros((noElements, 3)) 
# direction = np.zeros(3, 1)
# lambda = 
# steeringVector = np.exp(2j * np.pi * elementPos * direction/lambda)