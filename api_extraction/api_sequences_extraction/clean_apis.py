# These are calls that need to be filtered
register_calls = [
    'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp',
    'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'ebp', 'rbp'
]

noise_in_calls = [
    'offset', '+', 'arg_C', 'loc_', 'near ptr', 'dword ptr', 'dword_', 
    'word ptr', 'word_', 'Hash:', 'off_', '+var_', 'Code_execution', 
    'far ptr', 'byte_', 'funcs_', 'ImageList', '@', '__', '[', '(', 
    'entry_point_', 'cases', 'PK11_', '-', '*',
]

# These are strings and chars that are usually appended towards the end
remove_append = ['_', '$']

unique_apis = []  # Assuming this is populated elsewhere before being passed to functions


def clean_function_dictionary(function_dictionary, unique_api_list, function_addr_data):
    """
    This function performs the final check of matching extracted APIs with the unique API list,
    then it removes all empty keys in the dictionary.
    """
    clean_dict = {}
    clean_func_addr_dict = {}
    for block in function_dictionary:
        extracted_calls = function_dictionary[block]
        cleaned_calls = []
        cleaned_func_addr_data = []
        if extracted_calls:
            for call_idx in range(len(extracted_calls)):
                call = extracted_calls[call_idx]
                if call.startswith('0x'):
                    cleaned_calls.append(call)
                    cleaned_func_addr_data.append(function_addr_data[block][call_idx])
                    continue
                else:
                    for api in unique_api_list:
                        if call == api:
                            cleaned_calls.append(api)
                            cleaned_func_addr_data.append(function_addr_data[block][call_idx])
                            break
            function_dictionary[block] = cleaned_calls
            function_addr_data[block] = cleaned_func_addr_data
        clean_dict[block] = function_dictionary[block]
        clean_func_addr_dict[block] = function_addr_data[block]

    return clean_dict, clean_func_addr_dict


def additional_clean(api):
    """
    Further cleans a function call string by removing specific prefixes.
    """
    cleaned_api = str(api)
    take_prefix = ["; "]
    for string in take_prefix:
        if string in cleaned_api:
            cleaned_api = cleaned_api.split(string)[1]
    return cleaned_api


def clean_function(function_dictionary_old, unique_apis, function_addr_data):
    """
    Removes noise from call instructions.
    """
    function_dictionary = dict(function_dictionary_old)
    for block_addr in function_dictionary:
        function_calls = []
        func_addr_list = []
        # Iterating through all the calls
        for call_idx in range(len(function_dictionary[block_addr])):
            call = function_dictionary[block_addr][call_idx]
            if str(call) == "":
                continue
            flag = 0
            # Checking if the call is a register call
            if call in register_calls:
                flag = 1
                continue
            # Iterating through all the noise values
            try:
                for noise in noise_in_calls:
                    if noise in call:
                        flag = 1
                        #print(noise, call)
                        break
            except Exception as e:
                print(e)
                flag = 1
                function_dictionary[block_addr] = []
            if flag == 1:
                continue

            if '$' in call:
                new_call = additional_clean(call.split('$')[0])
                call = new_call

            elif '_' in call[-2:]:
                new_call = additional_clean(call.split('_')[0])                    
                call = new_call

            elif '; jumptable' in call:
                new_call = additional_clean(call.split('; jumptable')[0])
                call = new_call

            elif ' ; ' in call:
                
                new_call = additional_clean(call.split(' ; ')[0])
                if new_call in register_calls:
                    new_call = additional_clean(call.split(' ; ')[1])
                call = new_call

            elif '; Hash:' in call:
                print("additional call")
                new_call = additional_clean(call.split('; Hash:')[0])
                call = new_call
            elif "_0" in call:
                import IPython; IPython.embed(); exit()
                new_call = additional_clean(call.split("_0")[0])
                call = new_call
            
            if flag == 0:
                function_calls.append(call)
                func_addr_list.append(function_addr_data[block_addr][call_idx])

        function_dictionary[block_addr] = function_calls
        function_addr_data[block_addr] = func_addr_list

    function_dictionary, function_addr_data = clean_function_dictionary(
        function_dictionary, unique_apis, function_addr_data
    )
    return function_dictionary, function_addr_data
