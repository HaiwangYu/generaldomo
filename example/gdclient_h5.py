"""
Majordomo Protocol client example. Uses the mdcli API to hide all MDP aspects

Author : Min RK <benjaminrk@gmail.com>

"""

import sys
from generaldomo.client import Client
import zmq
import zio
import json
import numpy as np
import h5py

import h5_utils as h5u

def main():
    verbose = '-v' in sys.argv
    client = Client("tcp://localhost:5555", zmq.CLIENT, verbose)
    fn = "test/data-0.h5"
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    requests = 10
    for i in range(requests):
        img = h5u.get_hwc_img(fn, i%10, im_tags, [1, 10], [800, 1600], [0, 6000], 4000) # V
        try:
            label = json.dumps({"TENS":[{"dtype":'f',"part":1,"shape":img.shape,"word":4}]})
            # payload = np.array([[0, 1], [2, 3], [4, 5]], dtype='f')
            m = zio.Message(form='TENS', label=label, 
                 level=zio.MessageLevel.warning,
                 payload=[img.tobytes()])
            print(m)
            client.send(b"torch", m.toparts())
        except KeyboardInterrupt:
            print ("send interrupted, aborting")
            return
    
    out = h5py.File("out.h5", "w")
    count = 0
    while count < requests:
        try:
            reply = client.recv()
        except KeyboardInterrupt:
            break
        else:
            # also break on failure to reply:
            if reply is None:
                break
            m = zio.Message()
            m.fromparts(reply)
            label = json.loads(m.label)
            shape = label["TENS"][0]["shape"]
            payload = m._payload[0]
            mask = np.frombuffer(payload, dtype='f').reshape(shape)
            dset = out.create_dataset(name="/%d/mask"%count, shape=mask.shape, dtype='f', data=mask)
            print('reply:\n', mask.shape)
        count += 1
    print ("%i requests/replies processed" % count)

if __name__ == '__main__':
    main()