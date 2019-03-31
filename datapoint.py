class DataPoint:
    def __init__(self, x_co, y_co, cval):
        self.x = x_co
        self.y = y_co
        self.class_val = cval

    def __str__(self):
        return "{}, {}".format(self.x, self.y)

    def __repr__(self):
        return "{}, {}".format(self.x, self.y)
