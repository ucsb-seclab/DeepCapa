import networkx as nx

def caller_check(function_addr, block_addr, xrefs):
    flag = 0
    for call in xrefs:
        if call[0] == function_addr and call[1] == block_addr:
            flag = 1
            break
    return flag


def merge_xrefs(xref_dictionary, xref_1, function_addr):
    """ Merging the cross references of the functions.
        Args: 
            xref_dictionary (dict): Dictionary containing the cross references of the functions.
            xref_1 (list): List containing the cross references of the functions.
            function_addr (str): Address of the function.
        Returns:
            None
    """
    #There can be two cases, Fist, if the entire function is missing. Second, only few entries are missing
    if function_addr not in xref_dictionary:
        xref_dictionary[function_addr] = xref_1
    else:
        for entry in xref_1:
            if entry not in xref_dictionary[function_addr]:
                xref_dictionary[function_addr].append(entry)
            else:
                continue

def merge_call_addr_block_ordering(subset_id, call_addr_dictionary, function_addr, call_block_addr, sample_data, func_block_api_dictionary):
    #We have to iterate through both superset and subset
    block_addr = call_block_addr
    new_call_order = []
    
    superset_block_call = func_block_api_dictionary[function_addr][call_block_addr]
    call_addr_super = call_addr_dictionary[function_addr][call_block_addr]
    subset_block_call = sample_data[subset_id][function_addr]['call_dict'][call_block_addr]
    call_addr_sub = sample_data[subset_id][function_addr]['call_addr_dict'][call_block_addr]
    
    # We first check if one call block is the subset of another (it will make our work easier)
    if set(subset_block_call).issubset(set(superset_block_call)):
        new_call_order = superset_block_call
        call_addr_dictionary[function_addr][block_addr] = call_addr_super
        return new_call_order

    elif set(superset_block_call).issubset(set(subset_block_call)):
        new_call_order = subset_block_call
        call_addr_dictionary[function_addr][block_addr] = call_addr_sub
        return new_call_order
    else:
        # If either of the call list contains a call that is not present in other.
        # We have to perform merging
        instruction_addr_call_dict = dict()
        # Loop for superset

        for call_idx in range(len(superset_block_call)):
            #[2] because function and block addr will be same for each of these calls
            call_addr = int(call_addr_super[call_idx][2], 16)
            call_value = superset_block_call[call_idx]
            instruction_addr_call_dict[call_addr] = [call_value]
        
        # Loop for subset
        for call_idx in range(len(subset_block_call)):
            call_addr = int(call_addr_sub[call_idx][2], 16)
            call_value = subset_block_call[call_idx]
            if call_addr in instruction_addr_call_dict:
                continue
            else:
                instruction_addr_call_dict[call_addr] = [call_value]
        
        sorted_instruction_addr = sorted(instruction_addr_call_dict.keys())
        new_call_order = [instruction_addr_call_dict[instruction_addr][0] for instruction_addr in sorted_instruction_addr ]
        
        call_addr_dictionary[function_addr][block_addr] = [[function_addr, block_addr, hex(addr).lower()] for addr in sorted_instruction_addr]
        return new_call_order

'''
There exist two different functions merging_at_block_level_for_subset_graph and merging_at_block_level_for_duplicate_graph
because, the conditions and corner cases are handled differently in both the functions.
In the case of merging_at_block_level_for_subset_graph, one graph actually has more number of blocks(nodes) than the other one.
So we have to take that into account.
Ofc, we can come up with a way to comnine both merging_at_block_level_for_subset_graph and merging_at_block_level_for_duplicate_graph functions,
but thats for  a later time.
'''
def sort_based_on_address(call_list, xref_dictionary, block_addr, function_addr):
    xref_list_sorted = []
    for call in call_list:
        if "0x" in call:
            pass
        else:
            continue
    

def merging_at_block_level_for_subset_graph(current_saved_id, function_addr, sample_data, xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary, is_duplicate_flag):
    
    call_dict_1 = func_block_api_dictionary[function_addr]
    call_dict_2 = dict(sample_data[current_saved_id][function_addr]['call_dict'])
    G1 = nx.node_link_graph(graph_dictionary[function_addr])
    G2 = nx.node_link_graph(sample_data[current_saved_id][function_addr]['extracted_graph'])
    function_data_2 = sample_data[current_saved_id][function_addr]
    # We also have to merge the graph structure
    for node in G2.nodes:
        if node not in G1.nodes:
            G1.add_node(node)
    for edge in G2.edges:
        if edge not in G1.edges:
            G1.add_edge(*edge)
    new_call_list_dict = {}
    subset_id = current_saved_id
    
    for call_block_addr in call_dict_2:
        # When there exist a block in subset that is not present in superset
        if call_block_addr not in call_dict_1:
            new_call_list_dict[call_block_addr] = call_dict_2[call_block_addr]
            call_addr_dictionary[function_addr][call_block_addr] = function_data_2['call_addr_dict'][call_block_addr]
            continue

        # When calls in the block for both superset and subset are same. We dont need to make any changes
        if call_dict_2[call_block_addr] == call_dict_1[call_block_addr]:
            new_call_list_dict[call_block_addr] = call_dict_2[call_block_addr]
            continue
        else:
            call_list = merge_call_addr_block_ordering(subset_id, call_addr_dictionary, function_addr, call_block_addr, sample_data, func_block_api_dictionary)
            new_call_list_dict[call_block_addr] = call_list
            
    for block in new_call_list_dict:
        temp = list()
        temp_addr_list = list()
        if len(new_call_list_dict[block]) != len(call_addr_dictionary[function_addr][block]):
            print("\nValues dont match for blocks: at function: {}, block : {}, values: {} --> {} not present in superset: Block addr: {}, needs manual inspection. We currently take: {}".format(function_addr, block, new_call_list_dict[block], call_addr_dictionary[function_addr][block],  block, new_call_list_dict[block]))
            continue
            # print("addr for all calls not saved or vice versa")
            
        for call_index in range(len(new_call_list_dict[block])):
            if "0x" in new_call_list_dict[block][call_index]:
                ##Here call needs to be verified if present. 
                call_function_addr = new_call_list_dict[block][call_index]
                # This condition is needs to be checked, because there is a chance that block is not present in the new snapshot
                if block in sample_data[subset_id]:
                    if new_call_list_dict[block][call_index] in sample_data[subset_id]:
                        # We check if there exist an entry of call
                        if sample_data[subset_id][call_function_addr]['xrefs']:
                            # We merge the xrefs
                            merge_xrefs(xref_dictionary, sample_data[subset_id][call_function_addr]['xrefs'], call_function_addr)
                # Here we remove the call if it is not present in xref dictionary
                if call_function_addr not in xref_dictionary or not caller_check(function_addr, block, xref_dictionary[call_function_addr]):
                    continue
                else:
                    #print("xref found") 
                    temp.append(call_function_addr)
                    temp_addr_list.append(call_addr_dictionary[function_addr][block][call_index])
            else:    
                temp.append(new_call_list_dict[block][call_index])
                temp_addr_list.append(call_addr_dictionary[function_addr][block][call_index])
        if len(temp) > 0 and len(temp_addr_list) > 0:
            new_call_list_dict[block] = temp
            call_addr_dictionary[function_addr][block] = temp_addr_list

    func_block_api_dictionary[function_addr] = new_call_list_dict
    graph_dictionary[function_addr] = nx.node_link_data(G1)

def check_if_in_range(func_block_api_dictionary, function_addr):
    #If function address is present in dataset
    if function_addr in func_block_api_dictionary:
        return 1, 0
    else:
        for function in func_block_api_dictionary:
            start = int(function)
            end = int(func_block_api_dictionary['end'])
            #If the function address lies in the range of existing functions in the dataset
            if int(function_addr) >= start and int(function_addr) <= end:
                return 1, 0
            else:
                return 0, end
    pass
  
def merge_graphs(current_saved_id, function_addr, sample_data, xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary):    
    """Merging CFG of the same funtions from differnet memory snapshots if 
        there exist any API call or function call in the new snapshot that is absent in the old snapshot.

    Args: 
        current_saved_id (int): Current snapshot id.
        function_addr (str): Address of the function.
        sample_data (dict): Dictionary containing the data of the memory snapshots.
        xref_dictionary (dict): Dictionary containing the cross references of the functions.
        call_addr_dictionary (dict): Dictionary containing the call addresses of the functions.
        func_block_api_dictionary (dict): Dictionary containing the function block API calls.
        graph_dictionary (dict): Dictionary containing the networkx graphs of the functions.
    
    Returns:
        int: 1 if the merging is successful, 0 otherwise.
    
    """
    
    G1 = nx.node_link_graph(graph_dictionary[function_addr])
    G2 = nx.node_link_graph(sample_data[current_saved_id][function_addr]['extracted_graph'])
    
    # We have to make this assumption otherwise we can also use nx.is_isomorphic(G1, G2), but it is computationally expensive.
    # This holds true, if G1.nodes and G2.nodes are same then the edges are also same. 
    if set(list(G1.nodes())) == set(list(G2.nodes())):
        is_duplicate_flag = 1
        merging_at_block_level_for_subset_graph(current_saved_id, function_addr, sample_data, xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary, is_duplicate_flag)
        return 1
        
    else:
        #When G1 and G2 have different number of nodes
        #We check the call dict (call dict contains calls for each block)
        call_dict_1 = func_block_api_dictionary[function_addr]
        call_dict_2 = sample_data[current_saved_id][function_addr]['call_dict']
        # If the call dict are same. Which means the blocks containing calls are same in both the graph
        # We take the calls from the previous function (because it doesent matter)
        G1_nodes = list(G1.nodes())
        G2_nodes = list(G2.nodes())
        G1_subset_of_G2 = set(G1_nodes).issubset(set(G2_nodes))
        G2_subset_of_G1 = set(G2_nodes).issubset(set(G1_nodes))
        if G2_subset_of_G1 or G1_subset_of_G2:
            is_duplicate_flag = 0
            merging_at_block_level_for_subset_graph(current_saved_id, function_addr, sample_data, xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary, is_duplicate_flag)
            return 1
        else:
            call_dict1_nodes = [node for node in call_dict_1]
            call_dict2_nodes = [node for node in call_dict_2]
            call_dict1_subset_of_call_dict2 = set(call_dict1_nodes).issubset(set(call_dict2_nodes))
            call_dict2_subset_of_call_dict1 = set(call_dict2_nodes).issubset(set(call_dict1_nodes))
            if call_dict1_subset_of_call_dict2 or call_dict2_subset_of_call_dict1:
                is_duplicate_flag = 0
                merging_at_block_level_for_subset_graph(current_saved_id, function_addr, sample_data, xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary, is_duplicate_flag)
                return 1
            else:
                return 0
           
 

