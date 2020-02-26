"""Majordomo Protocol worker example.

Uses the mdwrk API to hide all MDP aspects

Author: Min RK <benjaminrk@gmail.com>
"""

import sys
from mdwrkapi import MajorDomoWorker
import zio
import json
import numpy as np

def main():
    verbose = '-v' in sys.argv
    worker = MajorDomoWorker("tcp://localhost:5555", b"torch", verbose)
    reply = None
    while True:
        request = worker.recv(reply)
        if request is None:
            break # Worker was interrupted
        m = zio.Message()
        m.fromparts(request)
        label = json.loads(m.label)
        shape = label["TENS"][0]["shape"]
        payload = m._payload[0]
        ret = np.frombuffer(payload, dtype='f').reshape(shape)
        print('input:\n', ret)
        ret = ret*10
        print('output:\n', ret)

        reply = zio.Message(form='TENS', label=m.label, 
              level=zio.MessageLevel.warning,
              payload=[ret.tobytes()]).toparts()



if __name__ == '__main__':
    main()