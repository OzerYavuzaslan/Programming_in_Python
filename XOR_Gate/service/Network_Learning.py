from service.Network_Calculation import NetworkCalculation
from service.Activation import Activation_Class

class NetworkLearning(object):
    def apply_backpropagation(instances, nodes, weights, learning_rate, 
                              activation_function, momentum):
        num_of_features = len(instances[0]) - 1
        
        for i in range(len(instances)):
            nodes = NetworkCalculation.apply_forward_propagation(nodes, weights, 
                                                                 instances[i], 
                                                                 activation_function)
            
            predicted_value = nodes[len(nodes) - 1].get_net_value()
            actual_value = instances[i][num_of_features]
            small_delta = predicted_value - actual_value
            nodes[len(nodes) - 1].set_small_delta(small_delta)
            
            for j in range(len(nodes) - 2, num_of_features, -1):
                target_index = nodes[j].get_index()
                sum_small_delta = 0
                
                for k in range(len(weights)):
                    if weights[k].get_from_index() == target_index:
                        affecting_weight = weights[k].get_value()
                        affecting_small_delta = 1
                        target_small_delta_index = weights[k].get_to_index()
                        
                        for m in range(len(nodes)):
                            if nodes[m].get_index() == target_small_delta_index:
                                affecting_small_delta = nodes[m].get_small_delta()
                        
                        newly_small_delta = affecting_weight * affecting_small_delta
                        sum_small_delta += newly_small_delta
                
                nodes[j].set_small_delta(sum_small_delta)
            
            previous_derivative = 0
            
            for j in range(len(weights)):
                weight_from_node_value = 0
                weight_to_node_delta = 0
                weight_to_node_value = 0
                weight_to_node_net_input = 0
                
                for k in range(len(nodes)):
                    if nodes[k].get_index() == weights[j].get_from_index():
                        weight_from_node_value = nodes[k].get_net_value()
                    
                    if nodes[k].get_index() == weights[j].get_to_index():
                        weight_to_node_value = nodes[k].get_net_value()
                        weight_to_node_net_input = nodes[k].get_net_input_value()
                        weight_to_node_delta = nodes[k].get_small_delta()
                
                """derivative = weight_to_node_delta * 
                (weight_to_node_value * (1 - weight_to_node_value)) * weight_from_node_value"""
                
                derivative = weight_to_node_delta * (weight_to_node_value * Activation_Class.derivative_calculation(activation_function, weight_to_node_value, weight_to_node_net_input) * weight_from_node_value)
                
                weights[j].set_value(weights[j].get_value() - learning_rate * derivative + momentum * previous_derivative)
                previous_derivative = derivative * 1
        
        return nodes, weights