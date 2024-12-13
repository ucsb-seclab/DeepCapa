import json
import pandas as pd


tech_names = {
    'T1071': 'Standard Application Layer Protocol',
    'T1055': 'Process Injection',
    'T1057': 'Process Discovery',
    'T1497': 'Virtualization/Sandbox Evasion',
    'T1134': 'Access Token Manipulation',
    'T1033': 'System Owner/User Discovery',
    'T1112': 'Modify Registry',
    'T1486': 'Data Encrypted for Impact',
    'T1082': 'System Information Discovery',
    'T1070': 'Indicator Removal',
    'T1095': 'Non-Application Layer Protocol',
    'T1083': 'File and Directory Discovery',
    'T1056': 'Input Capture',
    'T1518': 'Software Discovery',
    'T1543': 'Create or Modify System Process',
    'T1547': 'Boot or Logon Autostart Execution',
    'T1049': 'System Network Connections Discovery',
    'T1036': 'Masquerading',
    'T1568': 'Dynamic Resolution',
    'T1047': 'Windows Management Instrumentation',
    'T1552': 'Unsecured Credentials',
    'T1027': 'Obfuscated Files or Information',
    'T1014': 'Rootkit',
    'T1053': 'Scheduled Task / Job',
    'T1059': 'Command and Scripting Interpreter',
    'T1203': 'Exploitation for Client Execution',
    'T1485': 'Data Destruction',
    'T1564': 'Hide Artifacts',
    'T1562': 'Impair Defenses'
}

def create_table(result_path = "", results = {}):
    generic_set = ["T1036", "T1568", "T1047", "T1552", "T1027",
                "T1014", "T1053", "T1059", "T1203", "T1485",
                "T1564", "T1562"]
    api_dependent = [ 
                    "T1486", "T1082", "T1055", "T1056","T1083",
                    "T1070", "T1095", "T1071", "T1112", "T1547",
                    "T1497", "T1134", "T1518","T1049", "T1543", "T1033", "T1057"]
    
    
    api_dependent_tech_result = {"TechID":[], "Name": [], "TotalSamples": [], "TP": [], "TN": [], "FP": [], "FN": [], "Precision": [], "Recall": []}
    api_independent_tech_result = {"TechID": [], "Name": [], "TotalSamples": [], "TP": [], "TN": [], "FP": [], "FN": [],
                                 "Precision": [], "Recall": []}

    if result_path:
        with open(result_path, "r") as my_file:
            results = json.load(my_file)
    
    aggregated_api_dependent_result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    aggregated_api_independent_result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    techs_used = generic_set + api_dependent
    
    for tech in techs_used:
        if tech not in api_dependent:
            continue
        data = results[tech]
        # with open("{}{}/validation/testing_confusion_matrix.json".format(result_path,tech ), "r") as my_file:
        #     data = json.load(my_file)
        confusion_matrix = data
        TP = int(confusion_matrix["TP"])
        FP = int(confusion_matrix["FP"])
        TN = int(confusion_matrix["TN"])
        FN = int(confusion_matrix["FN"])
        total = TP + TN + FP + FN
        precision = "{:.2f}".format(round(((float(TP )/ (TP + FP))*100), 2))
        recall = "{:.2f}".format(round(((float(TP) / (TP + FN)) * 100), 2))
        aggregated_api_dependent_result["TP"] += int(TP)
        aggregated_api_dependent_result["TN"] += int(TN)
        aggregated_api_dependent_result["FP"] += int(FP)
        aggregated_api_dependent_result["FN"] += int(FN)
        api_dependent_tech_result["TechID"] += [tech]
        api_dependent_tech_result["Name"] += [tech_names[tech]]
        api_dependent_tech_result["TotalSamples"] += [int(total)]
        api_dependent_tech_result["TP"] += [int(TP)]
        api_dependent_tech_result["TN"] += [int(TN)]
        api_dependent_tech_result["FP"] += [int(FP)]
        api_dependent_tech_result["FN"] += [int(FN)]
        api_dependent_tech_result["Precision"] += [precision]
        api_dependent_tech_result["Recall"] += [recall]

    aggregated_precision_for_api_dependent = (float(aggregated_api_dependent_result["TP"]) / \
                                              (aggregated_api_dependent_result["TP"] + aggregated_api_dependent_result["FP"]))*100
    aggregated_recall_for_api_dependent = (float(aggregated_api_dependent_result["TP"]) / \
                                              (aggregated_api_dependent_result["TP"] + aggregated_api_dependent_result["FN"])) * 100
    api_dependent_tech_result["TechID"] += ["-"]
    api_dependent_tech_result["Name"] += ["API-Based Set"]
    api_dependent_tech_result["TotalSamples"] += [str(aggregated_api_dependent_result["TP"] + aggregated_api_dependent_result["TN"]+\
                                                     aggregated_api_dependent_result["FP"] + aggregated_api_dependent_result["FN"])]
    api_dependent_tech_result["TP"] += [aggregated_api_dependent_result["TP"]]
    api_dependent_tech_result["FP"] += [aggregated_api_dependent_result["FP"]]
    api_dependent_tech_result["TN"] += [aggregated_api_dependent_result["TN"]]
    api_dependent_tech_result["FN"] += [aggregated_api_dependent_result["FN"]]
    api_dependent_tech_result["Precision"] += [ "{:.2f}".format(round(aggregated_precision_for_api_dependent, 2))]
    api_dependent_tech_result["Recall"] += ["{:.2f}".format(round(aggregated_recall_for_api_dependent, 2))]



    for tech in techs_used:
        if tech not in generic_set:
            continue
        # with open("{}{}/validation/testing_confusion_matrix.json".format(result_path, tech), "r") as my_file:
        #     data = json.load(my_file)
        data = results[tech]
        confusion_matrix = data
        TP = int(confusion_matrix["TP"])
        FP = int(confusion_matrix["FP"])
        TN = int(confusion_matrix["TN"])
        FN = int(confusion_matrix["FN"])
        total = TP + TN + FP + FN
        precision = "{:.2f}".format(round(((float(TP) / (TP + FP)) * 100), 2))
        recall = "{:.2f}".format(round(((float(TP) / (TP + FN)) * 100), 2))
        aggregated_api_independent_result["TP"] += TP
        aggregated_api_independent_result["TN"] += TN
        aggregated_api_independent_result["FP"] += FP
        aggregated_api_independent_result["FN"] += FN
        api_independent_tech_result["TechID"] += [tech]
        api_independent_tech_result["Name"] += [tech_names[tech]]
        api_independent_tech_result["TotalSamples"] += [int(total)]
        api_independent_tech_result["TP"] += [int(TP)]
        api_independent_tech_result["TN"] += [int(TN)]
        api_independent_tech_result["FP"] += [int(FP)]
        api_independent_tech_result["FN"] += [int(FN)]
        api_independent_tech_result["Precision"] += [precision]
        api_independent_tech_result["Recall"] += [recall]

    aggregated_precision_for_api_independent = (float(aggregated_api_independent_result["TP"]) / \
                                              (aggregated_api_independent_result["TP"] + aggregated_api_independent_result[
                                                  "FP"])) * 100
    aggregated_recall_for_api_independent = (float(aggregated_api_independent_result["TP"]) / \
                                           (aggregated_api_independent_result["TP"] + aggregated_api_independent_result[
                                               "FN"])) * 100

    api_independent_tech_result["TechID"] += ["-"]
    api_independent_tech_result["Name"] += ["API-Independent Set"]
    api_independent_tech_result["TotalSamples"] += [
        str(aggregated_api_independent_result["TP"] + aggregated_api_independent_result["TN"] + \
            aggregated_api_independent_result["FP"] + aggregated_api_independent_result["FN"])]
    api_independent_tech_result["TP"] += [aggregated_api_independent_result["TP"]]
    api_independent_tech_result["FP"] += [aggregated_api_independent_result["FP"]]
    api_independent_tech_result["TN"] += [aggregated_api_independent_result["TN"]]
    api_independent_tech_result["FN"] += [aggregated_api_independent_result["FN"]]
    api_independent_tech_result["Precision"] += ["{:.2f}".format(round(aggregated_precision_for_api_independent, 2))]
    api_independent_tech_result["Recall"] += ["{:.2f}".format(round(aggregated_recall_for_api_independent, 2))]



    print("aggregated precison and recall for API dependent = {}    {}".format(aggregated_precision_for_api_dependent,aggregated_recall_for_api_dependent))
    print("aggregated precison and recall for API independent = {}    {}".format(aggregated_precision_for_api_independent,
                                                                               aggregated_recall_for_api_independent))
    try:
        api_dependent = pd.DataFrame.from_dict(api_dependent_tech_result)
        api_independent = pd.DataFrame.from_dict(api_independent_tech_result)
    except:
        print("error")
        import IPython
        IPython.embed()
        assert False
        for i in api_dependent_tech_result:
            print(i, len(api_dependent_tech_result[i]))

    api_dependent_csv = api_dependent.to_latex(index=False)
    api_independent_csv = api_independent.to_latex(index=False)

    with open("api_dependent_result.txt", "w") as my_file:
        api_dependent_csv = api_dependent_csv.replace("\n", " \\hline\n")

        my_file.write(api_dependent_csv)
    with open("api_independent_result.txt", "w") as my_file:
        api_independent_csv = api_independent_csv.replace("\n", " \\hline\n")
        my_file.write(api_independent_csv)

    all_tp = aggregated_api_dependent_result["TP"]  + aggregated_api_independent_result["TP"]
    all_tn = aggregated_api_dependent_result["TN"]  + aggregated_api_independent_result["TN"]
    all_fp = aggregated_api_dependent_result["FP"]  + aggregated_api_independent_result["FP"]
    all_fn = aggregated_api_dependent_result["FN"]  + aggregated_api_independent_result["FN"]
    total = all_tp + all_tn + all_fp + all_fn
    precision = round(((float(all_tp)*100)/(all_tp + all_fp)), 2)
    recall = round(((float(all_tp)*100)/(all_tp + all_fn)), 2)
    
    print("Total Precision = {} and Total Recall = {}".format(precision, recall))





if __name__ == "__main__":
    result_path = "results_path"
    create_table(result_path, {})
