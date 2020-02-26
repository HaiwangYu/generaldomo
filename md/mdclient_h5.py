"""
Majordomo Protocol client example. Uses the mdcli API to hide all MDP aspects

Author : Min RK <benjaminrk@gmail.com>

"""

import sys
from mdcliapi2 import MajorDomoClient
import zio
import json
import numpy as np

def main():
    verbose = '-v' in sys.argv
    client = MajorDomoClient("tcp://localhost:5555", verbose)
    requests = 1
    for i in range(requests):
        try:
            label = json.dumps({"TENS":[{"dtype":'f',"part":1,"shape":[3,2],"word":4}]})
            payload = np.array([[0, 1], [2, 3], [4, 5]], dtype='f')
            m = zio.Message(form='TENS', label=label, 
                 level=zio.MessageLevel.warning,
                 payload=[payload.tobytes()])
            print(m)
            client.send(b"torch", m.toparts())
        except KeyboardInterrupt:
            print ("send interrupted, aborting")
            return

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
            ret = np.frombuffer(payload, dtype='f').reshape(shape)
            print('reply:\n', ret)
        count += 1
    print ("%i requests/replies processed" % count)

if __name__ == '__main__':
    main()