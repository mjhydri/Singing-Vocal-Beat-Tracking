import torch
import dill

model1 = torch.load("/storage/moji/vocal_beat/results/experiment1/models/model-5.pt")
model=dill.loads(model1)
torch.save(model,"/storage/moji/vocal_beat/results/experiment1/models/model-55.pt", pickle_protocol=4)

if torch.cuda.is_available():
    return torch.load(io.BytesIO(b))
else:
    return torch.load(io.BytesIO(b), map_location=torch.device('cpu'))

torch.storage