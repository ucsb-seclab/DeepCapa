import pymongo
import gridfs
import json
import bson.json_util as json_util
import os
    


class ReduceDB:
    def __init__(self):
        # MongoDB
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage1"]
        self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()



    