class Optimizer(object):
    def __init__(self):
        pass

    def __call__(self, learning_rate):
        pass

class PassThrough(Optimizer):
    def __init__(self):
        pass

    def __call__(self,  val):
        return val
        
class Momentum(Optimizer):
    def __init__(self, alpha):
        self.alpha = alpha
        self.velocity = 0

    def __call__(self, val):
        self.velocity = self.alpha*self.velocity + val
        return self.velocity