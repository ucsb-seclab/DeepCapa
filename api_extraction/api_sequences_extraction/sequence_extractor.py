import os.path
#import sys
import random
import numpy as np
import time
from producer import Producer
import json
import sys

# We have to add coverage
class ApisExtraction:
    def __init__(self, sample_hash, sample_data, total_api_sequences, api_sequence_budget, output_dir=None):
        '''

        :param iterations: number of iterations to be performed
        :param num_of_apis: total number of APIs to take in each iteration
        :param extracted_api_path: path of the extracted_calls directory of the sample
        :param extracted_graph_path: path of the extracted picked control flow of each function
        '''
        self.iterations = total_api_sequences
        self.num_of_apis = api_sequence_budget
        self.hash = sample_hash
        self.sample_data = sample_data
        self.output_dir = output_dir
        self.func_addr_containting_apis = self.sample_data['function_addr_containing_apis'] 
        self.function_addr_containing_calls = self.sample_data['function_addr_containing_calls']
        self.func_block_api_dictionary = self.sample_data['func_block_api_dictionary']
        self.call_addr_dictionary  = self.sample_data['call_addr_dictionary']
        self.wt_matrix = {}
        self.index_of_block = {}
        self.block_addr_list = {}
        self.call_list_dictionary = {}
        self.xrefs = {}
        self.api_sequences = {}
        self.function_block_addr = {}
        self.function_call_state_stack = []
        self.function_stack = []
        self.function_block_weights = {}
        self.api_function_and_blocks_done = {}
        self.graphs = {}
        self.coverage_tracker = {}

        for func_addr in self.function_addr_containing_calls:
            if 'index_of_block' in self.sample_data['functions'][func_addr]:
               self.index_of_block[func_addr] =  self.sample_data['functions'][func_addr]['index_of_block']

            if 'wt_matrix' in self.sample_data['functions'][func_addr]:
                self.wt_matrix[func_addr] = np.array(self.sample_data['functions'][func_addr]['wt_matrix'])

            if 'block_addr_list' in self.sample_data['functions'][func_addr]:
                self.block_addr_list[func_addr] = self.sample_data['functions'][func_addr]['block_addr_list']

            if 'call_list_dict' in self.sample_data['functions'][func_addr]:
                self.call_list_dictionary[func_addr] = self.sample_data['functions'][func_addr]['call_list_dict']

            if 'xrefs' in self.sample_data['functions'][func_addr]:
                self.xrefs[func_addr] = self.sample_data['functions'][func_addr]['xrefs']
            if 'graph' in self.sample_data['functions'][func_addr]:
                self.graphs[func_addr] = self.sample_data['functions'][func_addr]['graph']
    
        for func_addr in self.func_addr_containting_apis:
            self.coverage_tracker[func_addr] = {}
            call_dict = self.func_block_api_dictionary[func_addr]
            for block_addr in call_dict:
                calls = list(call_dict[block_addr])
                chosen = []
                for call in calls:
                    if "0x" not in call:
                        chosen.append([0, call])
                if chosen:
                    self.coverage_tracker[func_addr][block_addr] = chosen

    '''
        the following function calculates the coverage
    '''
    def calculate_coverage(self):
        apis_not_covered = []
        total_api_calls = 0
        apis_covered = 0
        for function_addr in self.coverage_tracker:
            call_list = self.coverage_tracker[function_addr]
            for block in call_list:
                calls = call_list[block]
                for call in calls:
                    total_api_calls += 1
                    if call[0] == 1:
                        apis_covered += 1
                    else:
                        apis_not_covered.append([function_addr, block, call[1]])  
        coverage = round(float(apis_covered*100)/total_api_calls, 2)
        return coverage, total_api_calls, apis_covered, apis_not_covered

    
    '''
        genearting frequency distribution of each api in final sequence
    '''
    def generate_histogram(self):
        distribution = dict()
        for sequence_idx in self.api_sequences:
            sequence = self.api_sequences[sequence_idx]
            for api in sequence:
                if api not in distribution:
                    distribution[api] = 1
                else:
                     distribution[api] += 1

        sorted_distribution = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
        return sorted_distribution
    
    def get_functions_addr_list_based_on_coverage(self):
        """ Returns functions that have less than 100% API coverage
        Returns:
            functions_to_explore: The list of functions to explore
            block_pairs_to_start: The dictionary containing the block pairs to start    
        """
        functions_to_explore = []
        block_pairs_to_start = {}
        for function_addr in self.coverage_tracker:
            total_api_calls = 0
            apis_covered = 0
            call_list = self.coverage_tracker[function_addr]
            for block in call_list:
                total_block_api_calls = 0
                block_apis_covered = 0
                calls = call_list[block]
                for call in calls:
                    total_api_calls += 1
                    total_block_api_calls += 1
                    if call[0] == 1:
                        apis_covered += 1
                        block_apis_covered += 1
                if total_block_api_calls > 0:
                    if function_addr not in block_pairs_to_start:
                        block_pairs_to_start[function_addr] = []
                    block_coverage = round(float(block_apis_covered*100)/total_block_api_calls, 2)
                    if block_coverage < 99:
                        block_pairs_to_start[function_addr].append(block)
            coverage = round(float(apis_covered*100)/total_api_calls, 2)
            if coverage < float(99):
                functions_to_explore.append(function_addr)
        
        return functions_to_explore, block_pairs_to_start
    
    
    def pick_random_number(self, total_number_of_funcs):
        """ Replaces the important block(the ones containing calls) with our cleaned ones
        Args:
            total_number_of_funcs: The total number of functions
        Returns:
            rand_int: The random integer
        """

        rand_int = random.randint(0, total_number_of_funcs) % total_number_of_funcs
        return rand_int


    def graph_parser(self,function_addr, block_addr, sequence, start_time, count_recursion, call_list_dictionary, api_function_and_blocks):
        """ Parse the graph and extract the API calls
        Args:
            function_addr: The address of the function
            block_addr: The address of the block
            sequence: The sequence of APIs
            start_time: The time when the function was called
            count_recursion: The count of recursion
            call_list_dictionary: The dictionary containing the calls
            api_function_and_blocks: The list of function and blocks
        Returns:
            sequence: The sequence of APIs
        """

        # We have to return in case of function stack becomes greater than 5000 because of python
        
        if count_recursion > 19000:
            return sequence, count_recursion, api_function_and_blocks
        # We run an infinite loop
        while 1:
            # Checking if our sequence budget or the time budget has been reached
            current_time = time.time()
            if len(sequence) >= self.num_of_apis or abs(current_time-start_time)>10.0:
               break

            # Corner case
            if block_addr not in call_list_dictionary:
                break

            # If the code block contains either api for function call
            if call_list_dictionary[block_addr]:
                block_calls = call_list_dictionary[block_addr]
                # The block consists some call
                # api_call_idx only keeps track of API indexes and is only updated when API call is encountered
                api_call_idx = 0
                for call in block_calls:
                    
                    current_time = time.time()
                    if len(sequence) >= self.num_of_apis or abs(current_time - start_time) > 10.0:
                        break
                    # We check the nature of the call instruction (function call or API call)
                    # In case the instruction is a function call(calling another function, eg 0x401000...)
                    if "0x" == call[0:2]:
                        function_addr_new = call
                        # This is an optmizing step. We only parse the function call if the function contains atleast 1 function call or API call. 
                        if function_addr_new not in self.sample_data['function_addr_containing_calls']:
                            continue
                        # We initialize the new function call
                        if function_addr_new in self.wt_matrix and function_addr_new in self.index_of_block and \
                            function_addr_new in self.block_addr_list and function_addr_new in self.call_list_dictionary:
                            
                            next_block_addr = function_addr_new
                            self.function_stack.append(function_addr_new)
                            count_recursion += 1
                            if count_recursion > 19000:
                                return sequence, count_recursion, api_function_and_blocks
                            
                            sequence, count_recursion, api_function_and_blocks = self.graph_parser(function_addr_new,
                                                                          next_block_addr,
                                                                          sequence,
                                                                          start_time,
                                                                          count_recursion, 
                                                                          self.call_list_dictionary[function_addr_new],
                                                                          api_function_and_blocks)
                            if count_recursion > 19000:
                                return sequence, count_recursion, api_function_and_blocks
                        else:
                                continue
                    else:
                        # When the call is an API call we append it to the sequence
                        if call != "":
                            sequence.append(call)
                            api_function_and_blocks.append([function_addr,block_addr])
                            # Now we have to update the coverage_tracker
                            self.coverage_tracker[function_addr][block_addr][api_call_idx][0] = 1
                            api_call_idx += 1
                        else:
                            continue

            '''
            Once we are done with calls:
                1.0. We fetch the next node
                1.1 We take probabilistic_random to choose next block
                1.2 One we have the next node we make that as start node and continue(whithout calling recursion)
                2.1 If next node does not exist, We first check if the stack is empty,
                2.1.1 If the stack is empty it means we are in our root function and, we use xrefs to find next function:
                2.1.2.1 If xref exist we extract the function and block values and start execution
                2.1.2.2 If xref does not exist we jus return the current sequence of APIs
                2.2.1 If the stack is not empty it means we are in the nested function, in that case just return the current sequence of APIs
                
            '''
            current_block_index = self.index_of_block[function_addr][block_addr]
            # Fetching edge weights for selecting next node
            relative_weights = self.wt_matrix[function_addr][current_block_index]
            # If the current block is not the final block
            # It means there is alteast 1 non zero value in the wt matrix of current node
            if sum(relative_weights) > 0:
                #1.1, #1.2
                all_node_ids = [node_id for node_id in range(len(relative_weights))]
                
                '''
                random.choices use relative weights to make weighted decisions
                this approach is useful because in relative weight list, the nodes that do not have an edge has 0 as value
                output of random.choices is a list of size k
                '''
                # Totally random
                            
                next_block_id = (random.choices(all_node_ids,weights=relative_weights, k=1 ))[0]
                
                # Assigning new block address for the next iteration
                block_addr = self.block_addr_list[function_addr][next_block_id]
                # This is required in case call_list dictionary was taken from xrefs
                call_list_dictionary =self.call_list_dictionary[function_addr]
                continue

            else:
                
                ''' 
                2.2.1 This means the current blocks is one of the return blocks of the function.
                checking if the function stack has only 1 element, it means we are currently in the end of the root func
                '''
                if len(self.function_stack) > 1:
                    # If the function stack is greater than 1 it means we are in some child function, thus we just have to return
                    
                    break
                else:
                    #2.1.2.2
                    # We are in the root function, we now check the xrefs of the root function to continue the execution
                    
                    if function_addr not in self.xrefs:
                        # This function does not contain any callers
                        
                        break
                    else:
                        if len(self.xrefs[function_addr]) == 0:
                            # This function does not contain any callers
                            break
                        #2.1.2.1
                        #fetching the xrefs
                        xrefs = self.xrefs[function_addr]
                        xref_index = 0
                        # Randomly pick xref
                        xref_choices = len(xrefs)
                        xref_index = self.pick_random_number(xref_choices)
                        function_addr_new = xrefs[xref_index][0]
                        next_block_addr = xrefs[xref_index][1]
                        instruction_addr = xrefs[xref_index][2]
                        flag = 0
                        if next_block_addr not in self.call_addr_dictionary[function_addr_new]:
                            break

                        for call_idx in range(len(self.call_addr_dictionary[function_addr_new][next_block_addr])):
                                calls = self.call_addr_dictionary[function_addr_new][next_block_addr]
                                for call in calls:
                                    if call[2] == instruction_addr:
                                        call_number = call_idx
                                        flag = 1
                                        break
                        
                        if flag == 0:
                            break

                        # Checking if all the values for function_addr_new exists
                        if function_addr_new in self.wt_matrix and function_addr_new in self.index_of_block and \
                            function_addr_new in self.block_addr_list and function_addr_new in self.call_list_dictionary:
                            # Checking if next_block_addr is present
                            if next_block_addr not in self.call_list_dictionary[function_addr_new]:
                                break
                            # Creating a new call list dictionary
                            call_list_dictionary_new = dict(self.call_list_dictionary[function_addr_new])
                            calls_in_next_block = self.call_list_dictionary[function_addr_new][next_block_addr]
                            # This is to make sure traversal is continued after the call function_addr.
                            if call_number + 1 == len(calls_in_next_block):
                                calls_in_next_block = []
                            else:
                                calls_in_next_block = calls_in_next_block[call_number +1 : ]
                            
                            call_list_dictionary_new[next_block_addr] =  calls_in_next_block
                            self.function_stack = []
                            
                            self.function_stack.append(function_addr_new)
                            count_recursion += 1
                            sequence, count_recursion, api_function_and_blocks = self.graph_parser(function_addr_new,
                                                                          next_block_addr,
                                                                          sequence,
                                                                          start_time,
                                                                          count_recursion,
                                                                          call_list_dictionary_new,
                                                                          api_function_and_blocks)
                            if count_recursion > 19000:
                                return sequence, count_recursion, api_function_and_blocks
                        else:
                            break

        self.function_stack = self.function_stack[0:-1]
        return sequence, count_recursion, api_function_and_blocks


    def fetch_random_block(self, wt_matrix,block_addr_list):
        """ Fetches a random block from the list of blocks
        Args:
            wt_matrix: The weight matrix
            block_addr_list: The list of block addresses
        Returns:
            random_block_addr: The random block address
        """

        # This step is important because in the graph reduction stage we zeroed rows in wt matrix. 
        non_zero_indices = [index for index in range(len(wt_matrix)) if sum(wt_matrix[index]) > 0]
        # This means there exist no non zero indices(weird case)
        if len(non_zero_indices) == 0:

            random_block_addr = block_addr_list[0]

        else:
            random_index = self.pick_random_number(len(non_zero_indices))
            random_block_addr = block_addr_list[non_zero_indices[random_index]]
        return random_block_addr

    
    def sequence_extractor(self, function_addr, start_blocks_available=[]):
        """ Extracts the sequence of APIs
        Args:
            function_addr: The address of the function
            start_blocks_available: The list of blocks available
        Returns:
            apis: The list of APIs
        """

        # Picking a random function from the list of extracted functions with API calls
        # TODO: Make the code neater
        if function_addr in self.wt_matrix and function_addr in self.index_of_block and \
            function_addr in self.block_addr_list and function_addr in self.call_list_dictionary:
            # Fetching random block(remember we pick a random function and a random block within that function to start our execution)
            # Q. should we only randomly pick a block among all the blocks that contains api, or randomly pick any block??
            if len(self.wt_matrix[function_addr]):
                if len(self.wt_matrix[function_addr]) == 1:
                    block_addr = function_addr
                else:
                    if len(start_blocks_available) > 0:
                        block_addr_idx = self.pick_random_number(len(start_blocks_available))
                        block_addr = start_blocks_available[block_addr_idx]
                    else:
                        block_addr = self.fetch_random_block(self.wt_matrix[function_addr], self.block_addr_list[function_addr])
                start_time = time.time()
                sequence = []
                self.function_stack = []
                self.function_stack.append(function_addr)
                count_recursion = 0
                api_function_and_blocks = []

                apis, count_recursion, api_function_and_blocks_done = self.graph_parser(function_addr, block_addr, sequence, start_time, count_recursion, self.call_list_dictionary[function_addr], api_function_and_blocks)
            else:
                apis = []
                
            return apis, function_addr, block_addr, api_function_and_blocks_done
        else:
            return [], "", ""


    def api_extractor(self):
        """ Extracts the APIs call sequences
        Args:
            None
        Returns:
            Success or failure
        """

        """
        function_block_api_dictionary: processed funtction_addr->block->api-function_call dictionary
        function_path_dictionary: pickled control flow path of chosen functions

        ToDo: Generate paths for xrefs!!
        """
        sys.setrecursionlimit(20000)
        # We perform n iterations
        itr = 0
        start_time = time.time()
        start_funcs_available, start_blocks_available = list(self.get_functions_addr_list_based_on_coverage())
        
        if len(start_funcs_available) < 20:
            print("cannot take {}".format(self.hash))
            return 1
        
        # For the first iteration
        random_func_index = self.pick_random_number(len(start_funcs_available))
        function_addr = start_funcs_available[random_func_index]
        block_addr_available = start_blocks_available[function_addr]
        while itr < self.iterations:
            current_time = time.time()
            # Time budget for each sample
            if abs(current_time - start_time) > 600:
                break
                
            api_sequence, function_addr, block_addr, api_function_and_blocks_done = self.sequence_extractor(function_addr, block_addr_available)

            if len(api_sequence) > 1:
                
                self.api_sequences[str(itr)] = api_sequence
                self.function_block_addr[str(itr)] = [function_addr, block_addr]
                self.api_function_and_blocks_done[str(itr)] = api_function_and_blocks_done
                itr += 1
            else:
                api_sequence = []
                api_function_and_blocks_done = []
            
            start_funcs_available, start_blocks_available = list(self.get_functions_addr_list_based_on_coverage())
            if len(start_funcs_available) < 3:
                
                start_funcs_available = self.func_addr_containting_apis
                random_func_index = self.pick_random_number(len(start_funcs_available))
                function_addr = start_funcs_available[random_func_index]
                block_addr_available = []
               
            else:
                random_func_index = self.pick_random_number(len(start_funcs_available))
                function_addr = start_funcs_available[random_func_index]
                block_addr_available = start_blocks_available[function_addr]
                
        if len(self.api_sequences) > 10:
            
            coverage, total_api_calls, apis_covered, apis_not_covered = self.calculate_coverage()
            histogram = self.generate_histogram()
            print("hash: {}, coverage: {}".format(self.hash, coverage))

            if self.output_dir is None:
                p = Producer()
                message_data = {"hash": self.hash, "api_sequences": self.api_sequences,\
                                "sequence_function_and_block": self.function_block_addr, "api_function_and_blocks":self.api_function_and_blocks_done,\
                                "coverage":coverage, "total_api_calls": total_api_calls,"apis_covered": apis_covered,"apis_not_covered": apis_not_covered,\
                                "histogram": histogram}
                
                message = {"method": "push", "data":message_data }
                toSend = json.dumps(message)
                time.sleep(1)

                p.sendToQueue(toSend)
                p.killConnection()
                return 1
            
            else:
                message_data = {"hash": self.hash, "api_sequences": self.api_sequences,\
                                "sequence_function_and_block": self.function_block_addr, "api_function_and_blocks":self.api_function_and_blocks_done,\
                                "coverage":coverage, "total_api_calls": total_api_calls,"apis_covered": apis_covered,"apis_not_covered": apis_not_covered,\
                                "histogram": histogram}
                with open(os.path.join(self.output_dir, self.hash + ".json"), 'w') as f:
                    json.dump(message_data, f)
                return 1
        else:
            return 0

