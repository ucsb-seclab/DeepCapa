from pre_traversing import generate_block_api_dictionary
import networkx as nx
import numpy as np
import time
import sys
from sequence_extractor import ApisExtraction
class GraphReduction:
    def __init__(self, sample_hash, sample_data, api_sequences_budget=0, api_calls_budget=0, output_dir=None):
        ''' Class to reduce the graph of the function
        arguments:
                - sample_hash: hash of the sample
                - sample_data: data of the sample extracted from disassembler
                - api_sequences_budget: number of api sequences to extract
                - api_calls_budget: number of api calls to extract
                - output_dir: output directory
        '''
        
        self.hash = sample_hash
        self.sample_data = sample_data
        self.iterations = api_sequences_budget
        self.api_calls_budget = api_calls_budget
        self.output_dir = output_dir
        self.function_call_state_stack = []
        self.function_stack = []
        self.function_block_weights = {}
        self.reduced_function_graph_wt = {}
        self.wt_matrix = {}
        self.incoming_edges_dict = {}
        self.outgoing_edges_dict = {}
        self.call_list_dictionary = {}
        self.block_addr_list = {}
        self.index_of_block = {}
        self.function_snapshot_id_dictionary = {}
        self.visited = []
        

    
    '''
        Initializing the function
    '''
    def initialize_function(self, function_addr,function_block_api_dictionary, G ):
        #important nodes: blocks in the CFG of the function that either contain a function call or an API call
        important_nodes = function_block_api_dictionary[function_addr]
        # Adjoint Dictionary
        adj_dict = dict(G.adj)
        # initial initialization
        start_node = list(G.nodes)[0]
        end_nodes = []
        last_block = []
        #extracting leaf node
        #leaf node is the node that does not have any child nodes
        for node in G.nodes:
            if len(adj_dict[node]) == 0:
                end_nodes.append(node)
    
    
    def combine_dictionary(self, important_nodes_dictionary, origional_dictionary):
        """Replaces the important block(the ones containing calls) with our cleaned ones
        
        """
        call_list_dictionary = {}
        # Iterating through the blocks of CFG
        for block in origional_dictionary:
            # If the current block contains an API or function call
            if block in important_nodes_dictionary:
                # call_list_dictionary[block] = {}
                call_list_dictionary[block] = important_nodes_dictionary[block]
            else:
                # We check if the non api blocks only have function calls
                
                
                call_list_dictionary[block] = []

        return call_list_dictionary

    
    def fetch_incoming_and_outgoing_edge_dict(self, wt_matrix):
        """Fetch incoming node_id for each node and outgoing node_id for each node
        
        arguments:
                - wt_matrix: wt_matrix of CFG after removing edges
        return: 
                - new incoming_edge_dict for the updated graph
        """

        incoming_edge_dict = {}
        outgoing_edge_dict = {}
        end_edges = []
        # Parsing through each node (complexity n^2)--- BAD
        for node_index in range(len(wt_matrix)):
            count = 0
            # Checking if there are any outgoing edge from the current node
            for out_going_edge in range(len(wt_matrix)):
                # Checking if node(node_index) has an outgoing edge to node(out_going_edge)
                if wt_matrix[node_index][out_going_edge] > 0:
                    count += 1
                    if out_going_edge not in incoming_edge_dict:
                        #
                        incoming_edge_dict[out_going_edge] = [node_index]
                        
                    else:
                        incoming_edge_dict[out_going_edge] += [node_index]
                        
                    if node_index not in outgoing_edge_dict:
                        outgoing_edge_dict[node_index] = [out_going_edge]
                    else:
                        outgoing_edge_dict[node_index] += [out_going_edge]
            # This basically means there are no outgoing edges--->leaf node
        return incoming_edge_dict, outgoing_edge_dict


    def fetch_outgoing_ids(self,incoming_edges_dict, current_node_id ):
        outgoing_ids = []
        for index in incoming_edges_dict:
            if current_node_id in incoming_edges_dict[index]:
                outgoing_ids.append(index)
        return outgoing_ids


    def initializer(self, function_addr, function_block_api_dictionary, G):
        """Initializes the CFG, extracts start and leaf nodes, assign weights to each node.
        """
        important_blocks = function_block_api_dictionary[function_addr]
        original_blocks = dict(G.nodes.data())
        call_list_dictionary = self.combine_dictionary(important_blocks, original_blocks)
        #call_list_dictionary = function_block_api_dictionary[function_addr]
        
        """
            Steps:
            1. We extract the adjoint dictionary, it consist of Adjacency View
            2. We extract the start and end node(for assigning weigths)
            3. We generate adjacency matrix
            4. We assign edge weights based on the number of api/function calls present in the block
            5. We aggregae all edge weights for each block starting from end nodes
        """
        # 1
        adj_dict = dict(G.adj)
        # 2
        start_node = list(G.nodes)[0]

        end_nodes = []
        for node in G.nodes:
            if len(adj_dict[node]) == 0:
                end_nodes.append(node)

        # 3
        # #xtracting the adjacency matrix of current graph
        adjacency_matrix = nx.to_numpy_array(G)
        # Assigning index to each block. block_addr_list just contains function_addr of each block in CFG
        
        block_addr_list = [block_addr for block_addr in adj_dict]
        index_of_block = {block_addr_list[index]:index for index in range(len(block_addr_list)) }
        # 4
        if len(end_nodes) == 0:
            adjacency_matrix[len(G) - 1] = np.array([0. for i in range(len(G))])
            end_nodes = [block_addr_list[len(G) - 1]]
        wt_matrix = np.copy(adjacency_matrix)
        
        for block in block_addr_list:
            if len(call_list_dictionary[block]) > 0:
                for index in range(len(adjacency_matrix)):
                    #All the incoming edge will be updated
                    if adjacency_matrix[index][index_of_block[block]]:
                        #assigning weight to each block
                        wt_matrix[index][index_of_block[block]] += len(call_list_dictionary[block])
        # Initializing_block_weight_for_function
        
        func_addr = function_addr
        self.wt_matrix[func_addr] = np.copy(wt_matrix)
        self.function_block_weights[func_addr] = {block_addr:len(call_list_dictionary[block_addr]) for block_addr in block_addr_list}
        incoming_edges_dictionary, outgoing_edge_dict = self.fetch_incoming_and_outgoing_edge_dict(wt_matrix)
        self.outgoing_edges_dict[func_addr] = outgoing_edge_dict
        self.incoming_edges_dict[func_addr]= incoming_edges_dictionary
        self.index_of_block[func_addr] = index_of_block
        self.block_addr_list[func_addr] = block_addr_list
        self.call_list_dictionary[func_addr] = call_list_dictionary
        return index_of_block, block_addr_list, func_addr, call_list_dictionary, end_nodes, wt_matrix


    '''
        Reducing the CFG of the graph
        Approach:
            Keep the root and leaf nodes of the CFG intact
            For every other node, if the current node contains API or function call keep the node
            If the current node does not contain API or function call, remove the node and create a connection between the the patent and child of current node
    '''
    def graph_reducer_recursive(self, current_node_id, start, end, calllist_dict, 
                                                        node_index_list, function_addr, recursive_stack, start_time):
        cur_time = time.time()
        
        #We put a timer for 150seconds
        if cur_time - start_time > 60:
            return -1
        #print(current_node_id)
        '''
        :param start: The start node
        :param end: leaf node
        :param wt_matrix: wt_matrix of the entire graph
        :param calllist_dict: block_addr:[api/function call] dictionary for the current function
        :param node_index_list: list for index to block_addr mapping
        :param incoming_edges_dict: dictionary containing mapping of incoming edges for each block addr
        :return: wt matrix for the reduced graph
        '''
        
        #The idea is too keep the start and leaf nodes intact
        if current_node_id == start:
            all_outgoing_node_ids = list(self.outgoing_edges_dict[function_addr][current_node_id])
            for outgoing_node_id in all_outgoing_node_ids:
                if outgoing_node_id not in self.visited:
                    self.visited.append(outgoing_node_id)
                    ret= self.graph_reducer_recursive(outgoing_node_id, start, end, calllist_dict, 
                                                        node_index_list, function_addr, recursive_stack, start_time)
                    if ret == -1:
                        return -1
                else:
                    continue
            return 1
        #end--> leaf node
        if current_node_id in end or len(self.outgoing_edges_dict[function_addr][current_node_id]) == 0:
            #recursive_stack.remove(current_node_id)
            #print(recursive_stack)
            return 1
        current_block_addr = node_index_list[current_node_id]
        # this means it has a call
        all_outgoing_node_ids = list(self.outgoing_edges_dict[function_addr][current_node_id])
        if self.function_block_weights[function_addr][current_block_addr] > 0:
            for outgoing_node_id in all_outgoing_node_ids:
                if outgoing_node_id not in self.visited:
                    self.visited.append(outgoing_node_id)
                    ret= self.graph_reducer_recursive(outgoing_node_id, start, end, calllist_dict, node_index_list,\
                                                     function_addr, recursive_stack, start_time)
                    if ret == -1:
                        return -1
                    else:
                        continue
                else:
                    continue
        else:
            '''if the current block does not have either function or API call. 
            We remove unlink the current block, connect the incoming blocks to outgoing blocks'''

            #removing self loops, it is important to remove loops. Here self loop basically means, 
            #A->B, B->A, but we remove B node. So now the edge is A->A, it can be tricky to handle it.
                    
            for inc in self.incoming_edges_dict[function_addr][current_node_id]:
                if inc == current_node_id:
                    continue
                self.outgoing_edges_dict[function_addr][inc].remove(current_node_id)
                self.wt_matrix[function_addr][inc][current_node_id] = 0
                for out in self.outgoing_edges_dict[function_addr][current_node_id]:
                    if out == current_node_id:
                        continue
                    #if the incoming 
                    if out not in self.outgoing_edges_dict[function_addr][inc]:
                        self.outgoing_edges_dict[function_addr][inc] += [out]
                        self.incoming_edges_dict[function_addr][out] += [inc]
                        self.wt_matrix[function_addr][inc][out] = self.wt_matrix[function_addr][current_node_id][out]
            
            for out in self.outgoing_edges_dict[function_addr][current_node_id]:
                self.incoming_edges_dict[function_addr][out].remove(current_node_id)
            self.outgoing_edges_dict[function_addr][current_node_id] = []
            self.incoming_edges_dict[function_addr][current_node_id] = []
            self.wt_matrix[function_addr][current_node_id] = [0. for i in range(len(self.wt_matrix[function_addr]))]
            
            #making connection between parent and child of current node.
            for outgoing_id in all_outgoing_node_ids:
                if outgoing_id not in self.visited:
                    self.visited.append(outgoing_id)
                    ret = self.graph_reducer_recursive(outgoing_id, start, end, calllist_dict, node_index_list, function_addr, recursive_stack, start_time)
                    if ret == -1:
                        return -1
                    else:
                        continue
                else:
                    continue
        return 1


    def initialize_and_reduce_all_call_funcs(self, function_addr_containing_calls, graph_dictionary, function_block_api_dictionary):
        """ Initialize and reduce the graph of each function containing
        """
        #parsing through each function
        bef_time = time.time()
        ret = 1
        for func_addr in function_addr_containing_calls:
            
            G = nx.node_link_graph(graph_dictionary[func_addr])
            index_of_block, block_addr_list, func_addr, call_list_dictionary, end_nodes, wt_matrix = self.initializer(func_addr, function_block_api_dictionary, G)
            end_node_ids = [index_of_block[end_node_addr] for end_node_addr in end_nodes]
            
            recursive_stack = [0]
            #counter += 1
            start_timer = time.time()
            if len(end_node_ids) == len(wt_matrix):
                continue
            
            if len(end_node_ids) < 3 and end_node_ids[0] == 0:
                ret = 1
            else:                
                start = 0
                #fix this in initializer
                if int(sum([i for i in wt_matrix[0]])) == 0:
                    for i in range(len(wt_matrix)):
                        if sum(wt_matrix[i]) != 0:
                            start = i
            
                self.visited  = [start]
                sys.setrecursionlimit(5000)
                ret = self.graph_reducer_recursive(start, start, end_node_ids, call_list_dictionary, block_addr_list, func_addr, recursive_stack, start_timer)
            
            if ret == -1:
                print("returned -1")
                self.wt_matrix[func_addr] = wt_matrix
                print("sample: {}, function: {}".format(self.hash, func_addr))
        
        return ret


    def graph_reducer(self):
        """Reducing the CFG by first identifying the important blocks:
            - Identify blocks containing API calls, start , end (return or exit), and key control flow insturctions
            - Remove unimportant nodes
        """
        
        func_block_api_dictionary, func_addr_containting_apis, function_addr_containing_calls, xref_dictionary, graph_dictionary, call_addr_dictionary = generate_block_api_dictionary(self.sample_data)

        ret = self.initialize_and_reduce_all_call_funcs(function_addr_containing_calls, graph_dictionary, func_block_api_dictionary)
       
        
        if ret == 1:
            new_combined_dict = {}
            new_combined_dict['functions'] = {}
            new_combined_dict['hash'] = self.hash
            new_combined_dict['function_addr_containing_apis'] = func_addr_containting_apis
            new_combined_dict['function_addr_containing_calls'] = function_addr_containing_calls
            new_combined_dict['func_block_api_dictionary'] = func_block_api_dictionary
            new_combined_dict['call_addr_dictionary'] = call_addr_dictionary
            for func_addr in func_block_api_dictionary:
                new_combined_dict['functions'][func_addr] = {}
                if func_addr in self.index_of_block:
                    new_combined_dict['functions'][func_addr]['index_of_block'] = self.index_of_block[func_addr]
                if func_addr in self.wt_matrix:
                    new_combined_dict['functions'][func_addr]['wt_matrix'] = self.wt_matrix[func_addr].tolist()
                if func_addr in self.block_addr_list:
                    new_combined_dict['functions'][func_addr]['block_addr_list'] = self.block_addr_list[func_addr]
                if func_addr in self.call_list_dictionary:
                    new_combined_dict['functions'][func_addr]['call_list_dict'] = self.call_list_dictionary[func_addr]
                if func_addr in xref_dictionary:
                    #new_combined_dict['functions'][func_addr]['xrefs'] = combined_dict[func_addr]['xrefs']
                    new_combined_dict['functions'][func_addr]['xrefs'] = xref_dictionary[func_addr]
                if func_addr in graph_dictionary:
                     new_combined_dict['functions'][func_addr]['graph'] = graph_dictionary[func_addr]
            
            stage2_dict = new_combined_dict
            obj = ApisExtraction(stage2_dict['hash'], stage2_dict, self.iterations, self.api_calls_budget, output_dir = self.output_dir)
            ret = obj.api_extractor()
            
        return ret


    



