import os
import json
from clean_apis import clean_function
import networkx as nx
from graph_merging import merge_graphs
from graph_merging import check_if_in_range

def unique_apis_assigner():
    # Todo make this path relative!
    unique_apis_path = "./unique_apis.txt"
    with open(unique_apis_path, "r") as my_file:
        unique_apis = my_file.read().split("\n")
    return unique_apis

def find_total_funcs_with_calls(func_block_api_dictionary):
    total_funcs  =[]
    for function_addr in func_block_api_dictionary:
        flag_func_call = 0
        function_data = func_block_api_dictionary[function_addr]
        for block in function_data:
            block_data = function_data[block]
            if not block_data:
                continue
            else:
                total_funcs.append(function_addr)
                flag_func_call
                break
        if flag_func_call == 1:
            break
    return total_funcs

def find_total_funcs_with_api(func_block_api_dictionary):
    total_apis = 0
    api_funcs = []
    for function_addr in func_block_api_dictionary:
        flag_api_call = 0
        function_data = func_block_api_dictionary[function_addr]
        for block in function_data:
            block_data = function_data[block]
            for call in block_data:
                if '0x' not in call:
                    total_apis += 1
                    if function_addr not in api_funcs:
                        api_funcs.append(function_addr)
                    
    return api_funcs, total_apis

def find_new_apis(func_block_api_dictionary, functions):
    new_unique_apis = []
    unique_apis_path = "./unique_apis.txt"
    
    with open(unique_apis_path, "r") as my_file:
        current_unique_apis = my_file.read().split("\n")

    for function_addr in functions:
        flag = 0
        function_data = func_block_api_dictionary[function_addr]
        for block in function_data:
            block_data = function_data[block]
            for call in block_data:
                if '0x' not in call:
                    if call not in current_unique_apis:
                        new_unique_apis.append(call)

    return new_unique_apis

## NOTE: Make this more optimized
def find_api_functions(function_data):
    """returns API calls present in the function

    Args:
        function_data: dict

    Returns:
        list of API calls
    """
    apis = []
    for block in function_data:
        block_data =function_data[block]
        for call in block_data:
            if '0x' not in call:
                apis.append(call)
        
    return apis

def find_functions(function_data):
    """returns function calls present in the function

    Args:
        function_data: dict
    
    Returns:
        list of function calls
    """
    functions = []
    for block in function_data:
        block_data =function_data[block]
        for call in block_data:
            if '0x' in call:
                functions.append(call)
        
    return functions

def fetch_function_addr_path_dictionary(extracted_graph_path, function_snapshot_id_dictionary):
    all_paths = {}
    function_addr_path_dictionary = {}
    #Extracting pickle path for each function
    unique_snapshot_ids = []
    for function_addr in function_snapshot_id_dictionary:
        snapshot_id = function_snapshot_id_dictionary[function_addr]
        #fetching all paths for current snapshot ID
        if snapshot_id not in all_paths:
            all_files_for_snapshot = os.listdir("{}{}/".format(extracted_graph_path,function_snapshot_id_dictionary[function_addr] ))
            all_paths[snapshot_id] = all_files_for_snapshot
    for function_addr in function_snapshot_id_dictionary:
        snapshot = function_snapshot_id_dictionary[function_addr]
        picked_path = ""
        function = function_addr.split(".json")[0]
        for picked_function_path in all_paths[snapshot]:
            if function in picked_function_path and ".json" not in picked_function_path:
                picked_path = "{}{}/{}".format(extracted_graph_path, \
                                 snapshot, picked_function_path)
                break
        if len(picked_path) == 0:
            for snapshot in all_paths:
                for picked_function_path in all_paths[snapshot]:
                    if function in picked_function_path and ".json" not in picked_function_path:
                        picked_path = "{}{}/{}".format(extracted_graph_path, \
                                                       snapshot, picked_function_path)
                        break
        if picked_path:
            function_addr_path_dictionary[function_addr] = picked_path

    return function_addr_path_dictionary


def clean_func_block_api_dictionary(func_block_api_dictionary):
    cleaned_dict = {}
    for function_addr in func_block_api_dictionary:
        cleaned_dict[function_addr] = {}
        if func_block_api_dictionary[function_addr]:
            
            call_dict = func_block_api_dictionary[function_addr]
            for block_addr in call_dict:
                if call_dict[block_addr]:
                    cleaned_dict[function_addr][block_addr] = call_dict[block_addr]
                else:
                    continue
        else:
            continue
    return cleaned_dict


'''
combining functions and their meta-data from different snapshots
'''
def combine_sample_data(function_snapshot_id_dictionary, sample_data):
    combined_dict = {}
    for func_addr in function_snapshot_id_dictionary:
        snapshot_id = function_snapshot_id_dictionary[func_addr]
        combined_dict[func_addr] = sample_data['data'][snapshot_id][func_addr]
    return combined_dict


def generate_block_api_dictionary(sample_data):
    func_block_api_dictionary = {}
    # Getting list of all snapshots
    snapshots = [p for p in sample_data['data']]
    #print(snapshots)
    function_snapshot_id_dictionary = {}
    unique_apis = unique_apis_assigner()
    xref_dictionary = {}
    call_addr_dictionary = {}
    graph_dictionary = {}
    # Iterating though each process snapshot
    family_hash_dict = {}
    
    #This is only when performing baseline comparison with forecast
    # with open("forecast_sample_config.json", "r") as my_file:
    #     family_hash_dict = json.load(my_file)``
    for id in snapshots:
        #If xref_dictionary is empty
        if not xref_dictionary:
            #We create a new Xref dictionary if there does not exist one, we create that using the first snapshot that is processed 
            xref_dictionary = dict({function_addr: sample_data['data'][id][function_addr]['xrefs'] \
            for function_addr in sample_data['data'][id]})

        #Iterating through each function in snapshot
        for function_addr in sample_data['data'][id]:
            # Fetching calllist dictionary for the function function_addr
            
            function_data = sample_data['data'][id][function_addr]['call_dict']
            function_addr_data = sample_data['data'][id][function_addr]['call_addr_dict']
            #Cleaning API and function calls in calllist dict
            #We can actually remove cleaned_function_data because changes are already made in function_data by clean_function
            
            cleaned_function_data, cleaned_addr_data = clean_function(function_data, unique_apis, function_addr_data, function_addr)
            #When a new function is encountered for the first time
            if function_addr not in func_block_api_dictionary:
                func_block_api_dictionary[function_addr] = function_data
                
                graph_dictionary[function_addr] = sample_data['data'][id][function_addr]['extracted_graph']
                #Is this required?? why not just assign xrefs?
                function_snapshot_id_dictionary[function_addr] = id
                #Now we also add the missing xrefs
                for block_addr in function_data:
                    for function_call in function_data[block_addr]:
                        if "0x" in function_call and function_call not in xref_dictionary:
                            xref_dictionary[function_call] = sample_data['data'][id][function_call]['xrefs']
                        else:
                            continue
                # We also store its call_addr dict
                call_addr_dictionary[function_addr] = sample_data['data'][id][function_addr]['call_addr_dict']
            else:
                # If the function is already present
               
                apis_in_saved_func = find_api_functions(func_block_api_dictionary[function_addr])
                apis_in_new_func = find_api_functions(function_data)
                function_call_in_saved_func = find_functions(func_block_api_dictionary[function_addr])
                function_call_in_new_func = find_functions(function_data)
                previous_saved_id = function_snapshot_id_dictionary[function_addr]
                
                #We dont want to merge the graph if there exists no API or Function call in the cfg 
                # This is just an optimizing step
                if len(apis_in_saved_func) == 0 and len(apis_in_new_func) == 0 and len(function_call_in_saved_func) == 0 and len(function_call_in_new_func) == 0:
                    continue
                # We also dont have to perform the merging step if no new API call or function call is added, as everything else will be reduced anyway in the next step
                if len(apis_in_new_func) == len(apis_in_saved_func) and  len(function_call_in_new_func) == len(function_call_in_saved_func):
                    continue
                
                current_snapshot_id = id
                ret = merge_graphs(current_snapshot_id, function_addr, sample_data['data'], xref_dictionary, call_addr_dictionary, func_block_api_dictionary, graph_dictionary)
                continue
                
    cleaned_func_block_api_dictionary = clean_func_block_api_dictionary(func_block_api_dictionary)
    #print("done with the sample")
    func_addr_containting_apis, total_apis = find_total_funcs_with_api(cleaned_func_block_api_dictionary)
    function_addr_containing_calls = find_total_funcs_with_calls(cleaned_func_block_api_dictionary)
    
    return cleaned_func_block_api_dictionary, func_addr_containting_apis, function_addr_containing_calls, xref_dictionary, graph_dictionary, call_addr_dictionary