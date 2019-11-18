import sys
import zmq
from common import recv_image


import numpy as np
import argparse
import time

def socket_setup():
    host = "arvrobotserver-desktop"
    port = "5560"
    dest='draw'
    action='store_true'
    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("Collecting updates from server...")
    url = 'tcp://{}:{}'.format(host, port)
    print('Conecting to: {}'.format(url))
    socket.connect(url)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    return socket
