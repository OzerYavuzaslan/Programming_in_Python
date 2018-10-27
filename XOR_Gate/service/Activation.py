import math

class Activation_Class(object):
    def activate(activation_function, x):
        func = 0
        
        if activation_function == 1:
            func = 1 / (1 + math.exp(-x))
        elif activation_function == 2:
            func = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        elif activation_function == 3:
            func = math.log(1 + math.exp(x))
        
        return func
    
    def derivative_calculation(activation_function, y, x):
        derivative = 0
        
        if activation_function == 1:
            derivative = y * (1 - y)
        elif activation_function == 2:
            derivative = 1 - (y * y)
        elif activation_function == 3:
            derivative = 1 / (1 + math.exp(-x))
        
        return derivative