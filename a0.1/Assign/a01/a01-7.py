class Counter:
    """Counts how many times `increment` is called."""
    # TODO:
    # 1. In __init__, store an internal count variable starting at 0.
    # 2. Method increment(step: int = 1) adds `step` to the count.
    # 3. Method value() returns the current count.
    def __init__(self):
        self.count = 0
        self.step = 1

    def increment(self, i):
        self.count +=1
    
    def value(self):
        value = self.count
        return value
        

c = Counter()
for i in range(5):
    c.increment(i)
print(c.value())   # Expected: 5
