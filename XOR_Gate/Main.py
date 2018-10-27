from service.Network_Structure import NetworkStructure
from service.Network_Connection import NetworkConnection
from service.Network_Calculation import NetworkCalculation
from service.Network_Learning import NetworkLearning
from service.Network_Cost import NetworkCost
from service.Feature_Normalization import FeatureNormalization
import matplotlib.pyplot as plt
import copy

instances = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] #X1, X2, and result

num_of_features = len(instances[0]) - 1

momentum = 0.05
iteration = 15000
learning_rate = 0.3
loss_history = []
hidden_nodes = [3]

nodes = NetworkStructure.create_nodes(num_of_features, hidden_nodes)
#------------------------------------------------------------------------------
print("""Enter an Activation Function to use in the neural network:
    Sigmoid Function = 1
    Hyperbolic Function = 2
    Softplus Function = 3""")
activation_function = int(input("Enter the activation function number: "))
print("")
#------------------------------------------------------------------------------
instances = FeatureNormalization.normalize(instances, activation_function)
print("Normalized dataset: ", instances)
#------------------------------------------------------------------------------
print("------------------------------\nInitial Weights:\n")
weights = NetworkConnection.create_weights(nodes, num_of_features, hidden_nodes)

initial_weights = copy.deepcopy(weights)

print("------------------------------")
print("Iterations without Adaptive Learning Rate:\n")

for i in range(iteration + 1):
    nodes, weights = NetworkLearning.apply_backpropagation(instances, nodes, 
                                                           weights, 
                                                           learning_rate, 
                                                           activation_function, 
                                                           momentum)
    
    cost_value = NetworkCost.calculate_loss(instances, nodes, weights, 
                                            activation_function)
    
    if i % 500 == 0 or i == iteration:
        print("{}th Iteration - Cost Value: {}".format(i, cost_value))
        loss_history.append(cost_value)

plt.plot(loss_history, label = 'Gradient Descent')

print("------------------------------")
#---------------------------<Tuning The Network>-------------------------------

adaptive_loss_history = []
previous_cost_value = 0

print("\nIterations with Adaptive Learning Rate:\n")

for i in range(iteration + 1):
    nodes, initial_weights = NetworkLearning.apply_backpropagation(instances, 
                                                           nodes, 
                                                           initial_weights, 
                                                           learning_rate, 
                                                           activation_function, 
                                                           momentum)
    
    cost_value = NetworkCost.calculate_loss(instances, nodes, initial_weights, 
                                            activation_function)
    
    if cost_value < previous_cost_value:
        learning_rate += 0.1
    else:
        learning_rate -= (0.5 * learning_rate)
        
    previous_cost_value = cost_value * 1
    
    
    if i % 500 == 0 or i == iteration:
        print("{}th Iteration - Cost Value: {}".format(i, cost_value))
        adaptive_loss_history.append(cost_value)

plt.plot(adaptive_loss_history, label = 'Adaptive Learning')
#------------------------------------------------------------------------------

plt.legend(bbox_to_anchor = (1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()
#------------------------------------------------------------------------------

print("Backpropagation with Adaptive Learning Rate is over...")
print("------------------------------\n\nPREDICTIONS:")

for i in range(len(instances)):
    instance = instances[i]
    NetworkCalculation.apply_forward_propagation(nodes, weights, instance, 
                                                 activation_function)

    print("After Adaptive Learning, Actual (expected value):", 
          instance[len(instance) - 1], "- prediction:", 
          nodes[len(nodes) - 1].get_net_value())
