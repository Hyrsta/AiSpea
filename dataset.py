import json
import torch
from torch.utils.data import Dataset, DataLoader

class DialogueDataset(Dataset):
    """
    Custom Dataset for loading dialogue data from a JSON file.
    
    The JSON file is expected to have the following structure:
    {
      "metadata": { ... },
      "dialog": [
          {"role": "child", "text": "..."},
          {"role": "AI", "text": "..."},
          ...
      ],
      "labels": { ... }
    }
    
    Each sample returned will include the dialogue item along with the global metadata and labels.
    """
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): Path to the JSON file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get("metadata", {})
        self.dialog = self.data.get("dialog", [])
        self.labels = self.data.get("labels", {})
        self.transform = transform

    def __len__(self):
        return len(self.dialog)

    def __getitem__(self, idx):
        # Retrieve one dialogue item
        dialog_item = self.dialog[idx]
        
        # Create a sample that includes the dialogue item,
        # along with the metadata and labels.
        sample = {
            "metadata": self.metadata,
            "dialog_item": dialog_item,
            "labels": self.labels
        }
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_dialogue_data(file_path, batch_size=32, shuffle=True, transform=None):
    """
    Function for loading dialogue data using the DialogueDataset.
    
    Args:
        file_path (str): Path to the JSON file.
        batch_size (int, optional): Batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data.
        transform (callable, optional): Optional transform to apply to each sample.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = DialogueDataset(file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example usage:
if __name__ == "__main__":
    # Replace 'path_to_file.json' with the actual file path.
    data_loader = load_dialogue_data("path_to_file.json", batch_size=2, shuffle=False)
    
    for batch in data_loader:
        print(batch)
