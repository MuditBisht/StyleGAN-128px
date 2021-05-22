import torch

def NormalizeImage(img):
  img = torch.tensor(img.T, dtype=torch.float32)
  # mean = torch.mean(img, dim=(1,2))
  # rstd = torch.rsqrt( torch.mean(torch.square(img - mean[:, None, None]), dim=(1,2)) )
  mean = torch.tensor([127.5, 127.5, 127.5])
  rstd = torch.tensor([1/127.5, 1/127.5, 1/127.5])
  result = []
  for t, m, s in zip(img, mean, rstd):
    result.append((t - m) * s)
  tensor = torch.stack(result)
  return tensor



def UnNormalizeImage(img):
  mean = [127.5, 127.5, 127.5]
  std = [127.5, 127.5, 127.5]
  result = [ ]
  for t, m, s in zip(img, mean, std):
    result.append(t * s + m)
  tensor = torch.stack(result)
  return tensor