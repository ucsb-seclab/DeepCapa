import pika
import pymongo
import json
import gridfs
import bson.json_util as json_util
#import db_sample_hash_to_file_id
import atexit
import sys
import os
class Consumer:
    def __init__(self):
        # RabbitMQ
        credentials = pika.PlainCredentials('myuser', 'mypassword')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672, '/', credentials=credentials, heartbeat=0))
        self.channel = self.connection.channel()
        #import db_sample_hash_to_file_id
        self.db_sample_hash_to_file_id = {}
        self.channel.queue_declare(queue='malware')
        file_path = "hash_to_file_id/hash_to_file_id.json"
        if not os.path.exists('./hash_to_file_id'):
            os.makedirs('./hash_to_file_id')
        
        with open(file_path, "r") as my_file:
            self.db_sample_hash_to_file_id = json.load(my_file)
        
        print("loaded {} files".format(len(self.db_sample_hash_to_file_id)))
        def callback(ch, method, properties, body):
            
            this = self.byteArrayToJson(body)
            # Extract the extract the hash value (only field in the dicotionary)
            if this.get('method') == "end":
                with open(file_path, "wb") as my_file:
                    #json.dump(self.db_sample_hash_to_file_id, my_file)
                    db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
                    db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
                    my_file.write(db_sample_hash_to_file_id_str_b)
                    print("Written to {}".format(file_path))
                    #print("Written to {}".format("hash_to_file_id/hash_to_file_id.json"))
                    print("DONE")
                    
            else:
                if this.get('method') == "push":
                    print("pushed called")
                    self.insertData(this['data'])
                else:
                    print("update")
                    self.updateData(this['data'])


            #print(" [x] Received %r" % body)
        self.channel.basic_consume(queue='malware', on_message_callback=callback, auto_ack=True)
        print(' [*] Waiting for messages. To exit press CTRL+C')

        #MongoDB
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage1_64"]
        self.malware_db = db["stage1_64"]
        self.fs = gridfs.GridFS(db)
        # Connect to RabbitMQ and start consuming messages
        self.channel.start_consuming()


    def updateData(self,toInsert):
        #print("testing update data")
        sample_hash = toInsert['hash']
        if sample_hash in self.db_sample_hash_to_file_id:
            file_object_id = self.db_sample_hash_to_file_id[sample_hash]
            if self.fs.exists(file_object_id):
                #print("file {} Found".format(sample_hash))
                stored_data = json.load(self.fs.get(file_object_id))
                for snapshot_id in toInsert['data']:
                    if snapshot_id not in stored_data['data']:
                        stored_data['data'][snapshot_id] = toInsert['data'][snapshot_id]
                        self.fs.delete(file_object_id)
                        byte_data = bytes(json.dumps(stored_data), 'utf-8')
                        file_object_id = self.fs.put(byte_data, filename=toInsert['hash'])
                        self.db_sample_hash_to_file_id[sample_hash] = file_object_id
                        print("added {} to {} with object id: {}".format(snapshot_id, sample_hash, file_object_id))
                        
            else:
                print("{} not present in db_sample_hash_to_file_dict")

    def insertData(self,toInsert):
        # toInsert is now a byte array (from RabbitMQ) we will need to decode it to string and then to json
        data = bytes(json.dumps(toInsert), 'utf-8')
        file_object_id = self.fs.put(data, filename=toInsert['hash'])
        self.db_sample_hash_to_file_id[toInsert['hash']] = file_object_id
        print("{} inserted to DB with object id: {}".format(toInsert['hash'], file_object_id))
        
    def byteArrayToJson(self, byteArray):
        #t_string = byteArray.decode('utf8').replace("'", '"')
        t_string = byteArray.decode('utf8')
        try:
            t_json = json.loads(t_string)
        except Exception as e:
            print("exception: {}".format(e))
            import IPython
            IPython.embed()
            assert False
        return t_json


    def killConnection(self):
        with open("hash_to_file_id/hash_to_file_id.json", "wb") as my_file:
            db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
            db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
            my_file.write(db_sample_hash_to_file_id_str_b)
            print("Written to {}".format("hash_to_file_id/hash_to_file_id.json"))
            #json.dump(self.db_sample_hash_to_file_id, my_file)
        self.connection.close()
        self.client.close()


c = Consumer()

@atexit.register
def goodbye(self):
    with open("hash_to_file_id/hash_to_file_id.json", "wb") as my_file:
        #json.dump(self.db_sample_hash_to_file_id, my_file)
        db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
        db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
        my_file.write(db_sample_hash_to_file_id_str_b)
        print("Written to {}".format("hash_to_file_id/hash_to_file_id.json"))

    print("GoodBye.")
