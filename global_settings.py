import torch, os, warnings

warnings.filterwarnings("ignore")
os.system('clear')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')

if str(device) != 'cuda':
  print("Cuda was unable to start. Please consider restarting the computer ")
  exit()
