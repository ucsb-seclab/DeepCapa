import networkx as nx
import random
import numpy as np
import time
import pika
import json

class Producer:

    def __init__(self):
        credentials = pika.PlainCredentials('myuser', 'mypassword')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='stage2')


    def sendToQueue(self, message):
        #{somekey:string somevalye:pickle}
        self.channel.basic_publish(exchange='',
                                   routing_key='stage2',
                                   body=message)

    def killConnection(self):
        self.connection.close()
