"""Majordomo Protocol worker example.

Uses the mdwrk API to hide all MDP aspects

Author: Min RK <benjaminrk@gmail.com>
"""

import sys
from mdwrkapi import MajorDomoWorker
import zio
import json
import numpy as np
import torch

def main():
    verbose = '-v' in sys.argv
    worker = MajorDomoWorker("tcp://localhost:5555", b"torch", verbose)
    model = "ts-model/unet1024-l23-cosmic500-e50-gpu.ts"
    net = torch.jit.load(model)
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
        img = np.frombuffer(payload, dtype='f').reshape(shape)

        img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])) # hwc_to_chw
        img_tensor = img_tensor.cuda()
        with torch.no_grad():
          input = img_tensor.unsqueeze(0)
          print ("input.shape: ", input.shape)
          mask = net.forward(input).squeeze().cpu().numpy()
          print('mask.shape:\n', mask.shape)

        reply = zio.Message(form='TENS',
              label=json.dumps({"TENS":[{"dtype":'f',"part":1,"shape":mask.shape,"word":4}]}), 
              level=zio.MessageLevel.warning,
              payload=[mask.tobytes()]).toparts()



if __name__ == '__main__':
    main()