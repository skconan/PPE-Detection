import zmq
import random
import sys
import time
from common import send_image
import argparse
from PIL import Image


def main(args):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect('tcp://{}:{}'.format(args.host, args.port))
    print("Starting camera publisher!")
    i = 0
    delay = 1.0/30.0
    img = Image.open(args.input_img)
    while True:
        # TODO -- chage timestamp to nanosec
        send_image(socket, img, i, '{:6f}'.format(time.time()))
        i += 1
        time.sleep(delay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_img', help='input image')
    parser.add_argument('--port', default="5559", help='port')
    parser.add_argument('--host', default="localhost", help='host')
    args = parser.parse_args()
    main(args)
