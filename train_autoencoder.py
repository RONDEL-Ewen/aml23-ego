import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models import AutoEncoder
import pickle
import sys
import gc

def get_args():
    """
    Parse 'name=value' command line arguments.
    """

    args = {}

    for arg in sys.argv[1:]:

        key, value = arg.split('=')
        args[key] = value

    return args

def load_data(
    file_path: str
):
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data

def prepare_dataset(
    data
):
    
    emg_data = []
    rgb_frames = []

    for sample in data:
        emg_data.append(torch.cat((torch.tensor(sample['emg_left']), torch.tensor(sample['emg_right'])), dim = 1))
        rgb_frames.append(torch.stack(sample['RGB_frames']))

    emg_data = torch.stack(emg_data)
    rgb_frames = torch.stack(rgb_frames)

    return TensorDataset(rgb_frames, emg_data)

def train(
    model,
    device, 
    train_loader,
    optimizer,
    criterion,
    epochs = 100
):
    
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def test(
    model,
    device,
    test_loader
):
    
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss/len(test_loader)}")

def main():

    args = get_args()
    train_file = args.get('train_file', './S04/train_data.pkl')
    test_file = args.get('test_file', './S04/test_data.pkl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoEncoder().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    train_data = load_data(train_file)
    train_dataset = prepare_dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    #train(model, device, train_loader, optimizer, criterion)

    print(len(train_dataset))
    print(len(train_dataset[0][1][0]))
    """
    del train_data
    gc.collect()

    test_data = load_data(test_file)
    test_dataset = prepare_dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
    test(model, device, test_loader)

    del test_data
    gc.collect()
    """

if __name__ == "__main__":
    main()