import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1280)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def caption_generation(image_feature, model, tokenizer, device):
    text = "prefix prefix prefix prefix prefix:"
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(inputs["input_ids"].to(device), 40, prefix = image_feature, do_sample = False, num_beams=5)[0]
    output = tokenizer.decode(output)
    return output.split(':')[1].split('.')[0].lower()