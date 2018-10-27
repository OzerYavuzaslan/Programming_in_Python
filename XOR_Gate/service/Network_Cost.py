from service.Network_Calculation import NetworkCalculation

class NetworkCost(object):
    def calculate_loss(instances, nodes, weights, activation_function):
        cost_value = 0
        
        for i in range(len(instances)):
            instance = instances[i]
            nodes = NetworkCalculation.apply_forward_propagation(nodes, weights, 
                                                                 instance, 
                                                                 activation_function)
            predict = nodes[len(nodes) - 1].get_net_value()
            actual = instance[len(instance) - 1]
            loss = ((predict - actual) * (predict - actual))
            loss /= 2
            cost_value += loss
        
        cost_value /= len(instances)
        
        return cost_value