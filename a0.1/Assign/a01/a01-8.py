import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, q):
        distance = math.sqrt((self.x - q.x)**2 + (self.y - q.y)**2)
        return distance
    """
    A 2-D point supporting distance calculation.
    Usage:
        p = Point(3, 4)
        q = Point(0, 0)
        print(p.distance_to(q))  # 5.0
    """
    # TODO:
    # 1. Store x and y as attributes.
    # 2. Implement distance_to(other) using the Euclidean formula.


# Smoke test
p, q = Point(3, 4), Point(0, 0)
print(p.distance_to(q))
# assert round(p.distance_to(q), 1) == 5.0
