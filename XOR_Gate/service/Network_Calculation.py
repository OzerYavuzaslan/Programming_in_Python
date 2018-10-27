# import math
from service.Activation import Activation_Class

class NetworkCalculation(object):
    def apply_forward_propagation(nodes, weights, instance, activation_function):
        #transfer bias unit values as +1
        for i in range(len(nodes)):
            if nodes[i].get_is_bias_unit() == True:
                nodes[i].set_net_value(1)
    
        #tranfer instace features to input layer. activation function would not be applied for input layer.
        for i in range(len(instance) - 1): #final item is output of an instance, that's why len(instance) - 1 used to iterate on features
            var = instance[i]
            for j in range(len(nodes)):
                if i + 1 == nodes[j].get_index():
                    nodes[j].set_net_value(var)
    
        for i in range(len(nodes)):
            if nodes[i].get_level() > 0 and nodes[i].get_is_bias_unit() == False:
                net_input = 0
                net_output = 0
                target_index = nodes[i].get_index()
                for j in range(len(weights)):
                    if target_index == weights[j].get_to_index():
                        w_i = weights[j].get_value()
                        source_index = weights[j].get_from_index()
                        for k in range(len(nodes)):
                            if source_index == nodes[k].get_index():
                                x_i = nodes[k].get_net_value()
                                net_input += (x_i * w_i)
                                #print(xi," * ", wi," + ", end='')
                                break
                #iterate on weights end
                # net_output = 1 / (1 + math.exp(-net_input))
                net_output = Activation_Class.activate(activation_function, net_input)
                nodes[i].set_net_input_value(net_input)
                nodes[i].set_net_value(net_output)
        return nodes