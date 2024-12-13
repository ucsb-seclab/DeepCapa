import pymongo
import gridfs
import json
import bson.json_util as json_util
import os

        
    

class ApiExtractDB:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage2"]
        self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()

class APIExtractDBTimeShift:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:33099/")
        db = self.client["time_shift_stage_2"]
        self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()

class ReduceDB:
    def __init__(self):
        #MongoDB
        
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage1"]
        #self.malware_db = db["sample_metadata"]
        self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()


class ReduceDBx64:
    def __init__(self):
        #MongoDB
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage1_64"]
        #self.malware_db = db["sample_metadata"]
        self.fs = gridfs.GridFS(db)
        #db = self.client["sample_metadata"]
        #self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()

class ReduceDBTimeShift:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:33099/")
        db = self.client["time_shift_stage_1"]
        self.fs = gridfs.GridFS(db)
    def killConnection(self):
        self.client.close()

    