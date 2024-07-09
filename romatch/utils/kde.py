import torch


def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        #ngurupr1 - computing the norm for gaussian kernel e^(-x^2/2*std^2); where std is the width of the kernel a.k.a inner scale (std>0)
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1) #pdf?
    return density
