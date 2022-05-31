import torch
import torchvision
import onnx

batch_size =1 #2

device = "cuda" #if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

torch_model = models.vgg11(pretrained=True)
torch_model.avgpool = nn.AvgPool2d(kernel_size=1,stride=1)
torch_model.classifier[0] = nn.Linear(512,4096)
torch_model.classifier[6] = nn.Linear(4096,10)

torch_model.eval()

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

state_dict = torch.load('vgg11.pth')
torch_model.load_state_dict(state_dict, strict=True)

# Input to the model
x = torch.randn(batch_size, 3, 32, 32, requires_grad=False)

# Export the model
torch.onnx.export(torch_model,               
                  x,                 
                  "vgg11.onnx",   
                  export_params=True,       
                  opset_version=10,          
                  do_constant_folding=True, 
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},   
                                'output' : {0 : 'batch_size'}})





