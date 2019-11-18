import numpy
from PIL import Image
import io
import zmq

def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def send_image_data(socket, image_data, image_id, timestamp, flags=0):
    md = dict(
        id = image_id,
        timestamp = timestamp,
    )
    socket.send_json(md, flags|zmq.SNDMORE) # multi-part
    return socket.send(image_data.getvalue(), flags, copy=True)


def send_image(socket, image, image_id, timestamp, flags=0):
    md = dict(
        id = image_id,
        timestamp = timestamp,
    )
    socket.send_json(md, flags|zmq.SNDMORE) # multi-part
    return socket.send(image_to_byte_array(image), flags, copy=True)

# synchronous manner
def recv_image(socket, flags=0):
    print("Recieve Image")
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=True)
    print("Successfull")
    return md, Image.open(io.BytesIO(msg))
