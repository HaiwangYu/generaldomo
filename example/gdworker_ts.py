"""Majordomo Protocol worker example.

Uses the mdwrk API to hide all MDP aspects

Author: Min RK <benjaminrk@gmail.com>
"""

import sys
from generaldomo.worker import Worker
import zmq
import zio
import json
import numpy as np
import torch

def main():
    verbose = '-v' in sys.argv
    worker = Worker("tcp://localhost:5555", b"torch", zmq.CLIENT, verbose)
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
        label_tens = label["TENS"]["tensors"]
        label_meta = label["TENS"]["metadata"]
        shape = label_tens[0]["shape"]
        payload = m._payload[0]
        img = np.frombuffer(payload, dtype='f').reshape(shape)

        # numpy -> numpy
        img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])) # hwc_to_chw
        img_tensor = img_tensor.cuda()
        with torch.no_grad():
          input = img_tensor.unsqueeze(0)
          print ("input.shape: ", input.shape)
          mask = net.forward(input).squeeze().cpu().numpy()
          print('mask.shape:\n', mask.shape)

        label_tens = [{"dtype":'f',"part":1,"shape":mask.shape,"word":4}]
        reply = zio.Message(form='TENS',
              label=json.dumps({"TENS":{"tensors":label_tens, "metadata":label_meta}}), 
              level=zio.MessageLevel.warning,
              payload=[mask.tobytes()]).toparts()



if __name__ == '__main__':
    main()