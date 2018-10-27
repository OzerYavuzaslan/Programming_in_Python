from entity.Node import Node

class NetworkStructure(object):
    def create_nodes(num_of_features, hidden_nodes):
        
        nodes = []
        node_index = 0
        
        print("\nInput Layer:  ", end='')
        #---input layer---
        
        #bias unit
        node = Node()
        node.set_level(0)
        node.set_label("+1")
        node.set_index(node_index)
        node.set_is_bias_unit(True)
        nodes.append(node)
        node_index += 1
        print(node.get_label(), "\t", end='')
        #------------------------------
        #input layer nodes
        for i in range(num_of_features):
            node = Node()
            node.set_level(0)
            node.set_label("X" + str(i+1) + "\t")
            node.set_index(node_index)
            node.set_is_bias_unit(False)
            nodes.append(node)
            node_index += 1
            print(node.get_label(), "\t", end='')
        
        print("")
        #------------------------------
        #hidden layers
        for i in range(len(hidden_nodes)):
            print("Hidden Layer: ", end='')
            
            #bias unit
            node = Node()
            node.set_level(i + 1)
            node.set_label("+1")
            node.set_index(node_index)
            node.set_is_bias_unit(True)
            nodes.append(node)
            node_index += 1
            print(node.get_label(), "\t", end='')
            
            for j in range(hidden_nodes[i]):
                node = Node()
                node.set_level(i + 1)
                node.set_label(("N[" + str(i+1) + "][" + str(j+1) + "]"))
                node.set_index(node_index)
                node.set_is_bias_unit(False)
                nodes.append(node)
                node_index += 1
                print(node.get_label(), "\t", end='')
            print("")
            
        #Output Layer
        node = Node()
        node.set_level(1 + len(hidden_nodes))
        node.set_label("Output")
        node.set_index(node_index)
        node.set_is_bias_unit(False)
        nodes.append(node)
        node_index += 1
            
        print("\nOutput Layer: ", node.get_label())
        print("")
                
        return nodes