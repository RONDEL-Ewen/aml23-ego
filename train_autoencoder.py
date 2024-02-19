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
"""
def max_pool_rgb_frames(
    frames
):
    
    frames_pooled = F.max_pool3d(frames.unsqueeze(0), kernel_size=(5, 1, 1), stride=(5, 1, 1))
    return frames_pooled.squeeze(2)
"""
def prepare_dataset(
    data
):
    
    emg_data = []
    rgb_frames = []

    for sample in data:
        emg_data.append(torch.cat((torch.tensor(sample['emg_left'][-1], dtype=torch.float), torch.tensor(sample['emg_right'][-1], dtype=torch.float))))
        rgb_frames.append(torch.stack(sample['RGB_frames']).permute(1, 0, 2, 3))
        #frames = torch.stack(sample['RGB_frames'])
        #rgb_frames.append(frames.mean(dim=0))

    emg_data = torch.stack(emg_data)
    rgb_frames = torch.stack(rgb_frames)

    return TensorDataset(rgb_frames, emg_data)

def train(
    model,
    device, 
    train_loader,
    optimizer,
    criterion,
    epochs = 250
):
    
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0
        running_cos_sim = 0.0
        running_eucl_dist = 0.0
        num_batches = 0

        for inputs, targets in train_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            cos_sim = F.cosine_similarity(outputs, targets, dim=1).mean()
            eucl_dist = F.pairwise_distance(outputs, targets).mean()

            running_cos_sim += cos_sim.item()
            running_eucl_dist += eucl_dist.item()
            num_batches += 1

        # Compute the means for the epoch
        epoch_loss = running_loss / num_batches
        epoch_cos_sim = running_cos_sim / num_batches
        epoch_eucl_dist = running_eucl_dist / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: \n\tLoss: {epoch_loss} \n\tCosine Similarity: {epoch_cos_sim} \n\tEuclidean Distance: {epoch_eucl_dist}")

def test(
    model,
    device,
    test_loader
):
    
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    total_eucl_dist = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item()

            cos_sim = F.cosine_similarity(outputs, targets, dim=1).mean()
            eucl_dist = F.pairwise_distance(outputs, targets).mean()

            total_cos_sim += cos_sim.item()
            total_eucl_dist += eucl_dist.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_cos_sim = total_cos_sim / num_batches
    avg_eucl_dist = total_eucl_dist / num_batches

    print(f"Test: \n\tLoss: {avg_loss} \n\tAverage Cosine Similarity: {avg_cos_sim} \n\tAverage Euclidean Distance: {avg_eucl_dist}\n")

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
    #print(f"EMG data dimension: {train_dataset[0][1].size()}")
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    train(model, device, train_loader, optimizer, criterion)

    #print(len(train_dataset))
    
    del train_data
    gc.collect()

    test_data = load_data(test_file)
    test_dataset = prepare_dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
    test(model, device, test_loader)

    del test_data
    gc.collect()
    

if __name__ == "__main__":
    main()