import json
import os
import idaapi
import idautils
import idc
import networkx
import pickle
import logging
# The current code assumes you are using a mongo db database and rabbit mq server
import api_extraction.mongo_db.producer as producer
import time
from idautils import Segments
from idc import get_segm_name, get_segm_end
from idaapi import auto_wait, FF_CODE
from ida_bytes import get_flags
import ida_funcs

'''
TODO:
    1. Improve logging
    2. Currently we force IDA to create CFG for the code that is not part of the CFG. Come up with a more elegant solution.
    3. Optimize the Code further
'''

def explore_block(block_start_address, block_end_address, func_name_addr_dict, func_addr, unique_apis):
    """ Extracts function and API calls in sequence from a block.
    """
    unique_apis_set = set(unique_apis)  # Convert to set for faster lookup
    
    call_list = []
    call_list_addr = []
    address = block_start_address

    block_start_time = time.time()
    while address <= block_end_address:
        block_curr_time = time.time()
        # We give 5 seconds to process each block (which is already too much)
        if block_curr_time - block_start_time > float(5):
            break

        cmd = idc.generate_disasm_line(address, 0)
        # Check if the instruction is a jump, particularly a call
        if "jmp" in cmd:
            call_value = ""
            for call in func_name_addr_dict:
                if call in cmd:
                    call_value = func_name_addr_dict[call]
                    break
            if len(call_value) > 0:
                call_list.append(call_value)
                call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                address = idc.next_head(address)
                continue
            else:
                # Handle potential offset references
                if "off_" in cmd:
                    offset_address = cmd.split("off_")[1]
                    try:
                        offset_address_hex = int(offset_address, 16)
                        call_instruction = idc.generate_disasm_line(offset_address_hex, 0)
                        call = call_instruction.split(" ")[-1]
                        if call in unique_apis_set:
                            call_list.append(call)
                            call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                            address = idc.next_head(address)
                            continue
                    except Exception as e:
                        address = idc.next_head(address)
                        continue
                else:
                    # Parse the jump target and handle accordingly
                    jump_target = cmd.split("    ")
                    if len(jump_target) > 1:
                        jump_target = jump_target[1].strip()
                    else:
                        address = idc.next_head(address)
                        continue
                    if "_" in jump_target:
                        call = jump_target.split("_")
                        # This part has issues (potentially incomplete or fragile logic)
                        for call_idx in call:
                            # You can remove this check, and instead expand the API list in the api sequences extraction code
                            if call_idx in unique_apis_set:
                                call_list.append(call_idx)
                                call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                                break
                        address = idc.next_head(address)
                        continue
                    elif jump_target in unique_apis_set:
                        call_list.append(jump_target)
                        call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                        address = idc.next_head(address)
                        continue
                    else:
                        address = idc.next_head(address)
                        continue
        # Handle call instructions
        if "call" in cmd:
            call_value = ""
            for call in func_name_addr_dict:
                if call in cmd:
                    call_value = func_name_addr_dict[call]
                    call_list.append(call_value)
                    call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                    break
            if len(call_value) == 0:
                call_value = cmd.split("    ")
                if len(call_value) > 1:
                    call_value = call_value[1]
                    call_list.append(call_value)
                    call_list_addr.append([hex(func_addr), hex(block_start_address), hex(address)])
                    address = idc.next_head(address)
                    continue
                else:
                    address = idc.next_head(address)
                    continue

        address = idc.next_head(address)

    return call_list, call_list_addr

'''
This method extracts all the required metadata from IDA for a function by parsing the CFG.
'''
def get_nx_graph(ea, func_name_addr_dict, start_time, unique_apis):
    '''
    ea: Entry address of the function
    func_name_addr_dict: Will store the metadata of the function
    start_time: The starting time of the extraction
    '''
    call_dict = {}
    exception = {}
    call_addr_dict = {}
    # Initialize a NetworkX directed graph object
    nx_graph = networkx.DiGraph()
    # Get the function associated with the entry address
    func = idaapi.get_func(ea)
    flowchart = None
    try:
        # Extract the flowchart of the function from IDA
        flowchart = idaapi.FlowChart(func)
    except Exception as e:
        return "", "", "", -1, "{}: {}".format(e, "flowchart could not be generated")

    # Iterate through each block in the flowchart
    for block in flowchart:
        curr_time = time.time()
        # We set the time budget for processing each function to 90 seconds
        if curr_time - start_time > 90:
            return "", "", "", -1, "{}".format("function budget timeout reached, last processed block: {}".format(hex(block.start_ea)))
        
        # Fetch call instructions from the block
        call_list, call_list_addr = explore_block(block.start_ea, block.end_ea, func_name_addr_dict, ea, unique_apis)
        # Create a node for the block in the NetworkX graph
        nx_graph.add_node(hex(block.start_ea), **{'calllist': call_list})
        # Join predecessor blocks to the current block
        for pred in block.preds():
            nx_graph.add_edge(hex(pred.start_ea), hex(block.start_ea))
        # Join successor blocks to the current block
        for succ in block.succs():
            nx_graph.add_edge(hex(block.start_ea), hex(succ.start_ea))
        # If function or API call instructions were found, add them to the call_dict
        if call_list:
            call_dict[hex(block.start_ea)] = call_list
            call_addr_dict[hex(block.start_ea)] = call_list_addr
    
    # Extract cross-references (xrefs) to the current function
    x_refs = idautils.XrefsTo(ea)
    xref_list = []
    if x_refs:
        # Get the caller instruction address
        x_refs = [x.frm for x in x_refs]
        # Iterate all the references
        for ref in x_refs:
            # Fetch the function address where the instruction lies
            function_addr = idaapi.get_func(ref)
            if function_addr:
                fc = idaapi.FlowChart(function_addr)
                for block in fc:
                    if block.start_ea <= ref and block.end_ea > ref:
                        # Reference format: [caller_function_addr, caller_block_addr, caller_instruction_addr]
                        xref_list += [[hex(function_addr.start_ea), hex(block.start_ea), hex(ref)]]
    return nx_graph, call_dict, call_addr_dict, xref_list, 1, ""

'''
Extract metadata from a process snapshot.
'''

def create_function_if_needed(addr):
    auto_wait()
    ida_funcs.add_func(addr)
    auto_wait()

def analyze_text_segment():
    # Iterating over all segments
    flag = 0
    for seg in Segments():
        segment_name = get_segm_name(seg)
        # This is because for some reason the name of the .text segment is sometimes messed up
        if '.text' in segment_name.lower() or 'code.' in segment_name.lower():
            flag = 1
            seg_start = seg
            seg_end = get_segm_end(seg_start)
            for addr in range(seg_start, seg_end):
                if get_flags(addr) & FF_CODE:
                    if not ida_funcs.get_func(addr):
                        create_function_if_needed(addr)
            break

def cfg_extractor(snapshot_id, sample_hash):
    #Initializing mongodb object
    #p = producer.Producer()
    p = producer.ProducerTimeShift()
    exception_log = {}
    snapshot_data = dict()
    try:
        analyze_text_segment()
        #fetching the list of all function addresses from snapshot
        function_list = [f for f in idautils.Functions()]
        func_name_addr_dict = dict()
        
        #for wiseau
        with open('./unique_apis.txt', "r") as my_file:
            unique_apis = my_file.read().split("\n")

        # Why are there 2 loops?
        for func in function_list:
            func_name_addr_dict[idc.get_func_name(func)] = hex(func)
        
        start_time = time.time()
        for func_addr in function_list:
            curr_time = time.time()
            #we set the time budget as 180seconds
            if curr_time - start_time > 180:
                #handling when there is a timeout
                if len(snapshot_data) > 0:
                    #writing exception for debugging purposes, this sub-optmial, needs to be refined
                    if exception_log:
                        exception_log[0]= {"exception_name": "timeout", "last_function_processed":str(func_addr)}
                        snapshot_data["exception_log"] = exception_log
                    else:
                        exception_id = len(exception_log)
                        exception_log[id]= {"exception_name": "timeout", "last_function_processed":str(func_addr)}
                        snapshot_data["exception_log"] = exception_log
                    message = {"method": "update", "data": {"hash":sample_hash, "data": {"snapshot{}".format(snapshot_id): snapshot_data}}}
                    toSend = json.dumps(message)
                    p.sendToQueue(toSend)
                    p.killConnection()
                    return -1, "Process terminated due to timeout"
                else:
                    return -1, "Empty Dictionary"

            #extracting metadata for the function func_addr        
            nx_graph, call_dict, call_addr_dict, xref_list, ret, exception = get_nx_graph(func_addr, func_name_addr_dict, start_time, unique_apis)
            #ret will be -1 if function get_nx_graph was terminated due to an exception
            if ret == -1:
                if len(snapshot_data) > 0:
                    if len(snapshot_data) > 0:
                        if exception_log:
                            exception_log[0]= {"exception_name": exception, "last_function_processed":str(func_addr)}
                            snapshot_data["exception_log"] = exception_log
                        else:
                            exception_id = len(exception_log)
                            exception_log[id]= {"exception_name": exception, "last_function_processed":str(func_addr)}
                            snapshot_data["exception_log"] = exception_log
                        message = {"method": "update", "data": {"hash":sample_hash,"data": {"snapshot{}".format(snapshot_id): snapshot_data}}}
                        toSend = json.dumps(message)
                        p.sendToQueue(toSend)
                        p.killConnection()
                        return -1, "Process terminated due to timeout"
                    else:
                        return -1, "Empty Dictionary"
                return -1, str(exception)
            #converting the graph to a format that can be used in the later stages
            encoded_graph = networkx.node_link_data(nx_graph)
            snapshot_data[hex(func_addr)] = {"call_dict": call_dict, "call_addr_dict":call_addr_dict, "xrefs": xref_list, "extracted_graph": encoded_graph}
            
        message = {"method": "update", "data": {"hash":sample_hash,"data": {"snapshot{}".format(snapshot_id): snapshot_data}}}
        toSend = json.dumps(message)
        p.sendToQueue(toSend)
        p.killConnection()
        return 1, ""


    except Exception as e:
        if exception_log:
            exception_log[0]= {"exception_name": "{}-----{}".format(str(e), "errored out in cfg extractor function")}
            snapshot_data["exception_log"] = exception_log
        else:
            exception_id = len(exception_log)
            exception_log[id]={"exception_name": "{}-----{}".format(str(e), "errored out in cfg extractor function")}
            snapshot_data["exception_log"] = exception_log
        
        message = {"method": "update", "data": {"hash":sample_hash,"data": {"snapshot{}".format(snapshot_id): snapshot_data}}}
        toSend = json.dumps(message)
        p.sendToQueue(toSend)
        p.killConnection()
        return -1, str(e)
