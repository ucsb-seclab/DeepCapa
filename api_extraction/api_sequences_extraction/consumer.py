import pika
import pymongo
import json
import gridfs
import bson.json_util as json_util
#import db_sample_hash_to_file_id
import atexit
import sys
class Consumer:
    def __init__(self):
        # RabbitMQ
        credentials = pika.PlainCredentials('myuser', 'mypassword')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672, '/', credentials=credentials, heartbeat=0))
        self.channel = self.connection.channel()
        #import db_sample_hash_to_file_id
        self.db_sample_hash_to_file_id = {}
        self.channel.queue_declare(queue='stage2')
        
        def callback(ch, method, properties, body):
            
            this = self.byteArrayToJson(body)
            # Extract the extract the hash value (only field in the dicotionary)
            if this.get('method') == "end":
                print("Writing to json")
                
                with open("hash_to_file_id/hash_to_file_id_stage_2.json", "wb") as my_file:
                    #json.dump(self.db_sample_hash_to_file_id, my_file)
                    db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
                    db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
                    my_file.write(db_sample_hash_to_file_id_str_b)
                    print("Written to {}".format("hash_to_file_id/hash_to_file_id_stage_2.json"))
                    print("DONE")
                    sys.exit()
            else:
                if this.get('method') == "push":
                    print("pushed called")
                    self.insertData(this['data'])
                else:
                    print("method not found")
                    


            #print(" [x] Received %r" % body)
        self.channel.basic_consume(queue='stage2', on_message_callback=callback, auto_ack=True)
        print(' [*] Waiting for messages. To exit press CTRL+C')

        #MongoDB
        self.client = pymongo.MongoClient("mongodb://mitrepredict:mitrepredict_mongo_db@localhost:27017/")
        db = self.client["stage2"]
        self.malware_db = db["stage2"]
        self.fs = gridfs.GridFS(db)
        # Connect to RabbitMQ and start consuming messages
        self.channel.start_consuming()

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
            
        return t_json


    def killConnection(self):
        with open("hash_to_file_id/hash_to_file_id_stage_2.json", "wb") as my_file:
            db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
            db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
            my_file.write(db_sample_hash_to_file_id_str_b)
            print("Written to {}".format("hash_to_file_id/hash_to_file_id_stage_2.json"))
            #json.dump(self.db_sample_hash_to_file_id, my_file)
        self.connection.close()
        self.client.close()


c = Consumer()

@atexit.register
def goodbye(self):
    with open("hash_to_file_id/hash_to_file_id_stage_2.json", "wb") as my_file:
        #json.dump(self.db_sample_hash_to_file_id, my_file)
        db_sample_hash_to_file_id_str = json_util.dumps(self.db_sample_hash_to_file_id)
        db_sample_hash_to_file_id_str_b = bytes(db_sample_hash_to_file_id_str, 'utf-8')
        my_file.write(db_sample_hash_to_file_id_str_b)
        print("Written to {}".format("hash_to_file_id/hash_to_file_id_stage_2.json"))

    print("GoodBye.")
