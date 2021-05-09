#!/usr/bin/python3

import socket
import _thread
import json
import random
import sys

ip = "127.0.0.1"
if (len(sys.argv)>1):
    port = int(sys.argv[1])
else:
    port = 33333

bumpers = ["ANY", "ASSIGNED", "FALSIFIED",
           "FALSIFIED_AND_PROPAGATED", "EFFECTIVE", "EFFECTIVE_AND_PROPAGATED"]
bumpStrategies = ["ALWAYS_ONE", "DEGREE",
                 "COEFFICIENT", "RATIO_DC", "RATIO_CD"]


def listen_client(client, id):
    random.seed(12345)
    while True:
        print(f'-- waiting for data for client {id} --')
        data = client.recv(4096)
        if len(data) == 0:
            print(f'client {id} disconnected, stopping')
            return
        request = data.decode("utf-8")
        print(f'Data received from client {id} : ', request)
        response = {"bumper": bumpers[random.randrange(len(bumpers))], "bumpStrategy": bumpStrategies[random.randrange(len(bumpStrategies))]}
        print(f'Sending the json response %s to client',response)
        client.sendall(json.dumps(response).encode('utf-8'))
        client.send("\n".encode('utf-8'))


# Using TCP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    # Listening for clients
    s.bind((ip, port))
    s.listen()
    clientid = 1

    try:
        while True:
            print(f'Fake DAC program waiting for {clientid}')
            client, address = s.accept()
            _thread.start_new_thread(listen_client, (client, clientid))
            clientid += 1
    except KeyboardInterrupt:
        print(' ... ok, the server has been stopped  ...')
