# API Call Extraction

To extract API call sequences from CFGs of individual memory snapshots in 3 steps: CFG Merging, CFG Reduction, and CFG exploration

## CFG Merging:
In this step we merge the CFG from every individual snapshot to create a unified CFG of the process. 

## CFG Reduction:
This step enables us to to perform an efficient eploration by removing all the redundant blocks from the Merged CFG. 
In this step we remove all the blocks from the CFG that: 
- Do not correspond to an internal API call or external function call invocation
- That are not important to maintain the control flow integrity.


## CFG Exploration:
In this step we perform multiple random walks on the CFG to extract API call sequences. 



## Note:
While developing DEEPCAPA, in order to scale our data processing pipeline, we used MongoDB and Rabbit-MQ to manage our DB.
