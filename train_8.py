import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchinfo
from tqdm import tqdm


# Custom Concat/Addition module
class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat((x, y), dim=1)


# Create the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.norm4 = nn.BatchNorm2d(256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.concat = Concat()
        self.fc3 = nn.Linear(2, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 256)
        self.dropout4 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 2, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(256, 1)

    def forward(self, x, num, target_pixel):
        x = self.norm1(self.gelu(self.conv1(x)))
        x = self.norm2(self.gelu(self.conv2(x)))
        x = self.max_pool(x)
        x = self.norm3(self.gelu(self.conv3(x)))
        x = self.norm4(self.gelu(self.conv4(x)))
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        y = self.concat(num.unsqueeze(1), target_pixel.unsqueeze(1))
        y = self.dropout3(self.relu(self.fc3(y)))
        y = self.dropout4(self.relu(self.fc4(y)))

        x = self.concat(x, y)

        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.output(x)
        x = self.sigmoid(x)

        return x


class MyMNISTTrainDataset(Dataset):
    def __init__(self):
        super(MyMNISTTrainDataset, self).__init__()
        self.dataset = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset) * 784

    def __getitem__(self, idx):
        dataset_size = len(self.dataset)
        pixel_index = idx // dataset_size
        img_index = idx % dataset_size

        img_tensor, num = self.dataset[img_index]

        # Get the pixel at the specified index as the target label
        target_pixel = img_tensor.view(-1)[pixel_index].detach().clone()

        # Create a mask tensor with 784 elements
        mask = torch.arange(784) >= pixel_index
        # Apply the mask to the image tensor
        img_tensor.view(-1)[mask] = -1.0

        return (
            img_tensor,
            torch.tensor(num / 9.0, dtype=torch.float32),
            torch.tensor(pixel_index / 784.0, dtype=torch.float32),
            target_pixel,
        )


if __name__ == "__main__":
    dataset = MyMNISTTrainDataset()

    img, num, index, target = dataset[0]
    print(f"Image shape: {img.shape}, Number: {num}, Index: {index}, Target: {target}")

    # print(f"Image:\n{data[0]}")
    # print(f"Hot position:\n{data[1]}")
    # for i in range(2, 12):
    #     print(f"Number channel {i-2}:\n{data[i]}")

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )

    # # Print first batch to test the dataset
    # for data, target in dataloader:
    #     print(f"Data: {data}, Target: {target}")
    #     break

    # Get type of the first batch
    for img, num, index, target in dataloader:
        # print(f"Data type: {data.dtype}, Target type: {target.dtype}")
        print(
            f"Image shape: {img.shape}, Number shape: {num.shape}, Index shape: {index.shape}, Target shape: {target.shape}"
        )
        print(
            f"Image dtype: {img.dtype}, Number dtype: {num.dtype}, Index dtype: {index.dtype}, Target dtype: {target.dtype}"
        )
        break

    # Create an instance of the model
    model = SimpleCNN()

    # Print a summary of the model
    torchinfo.summary(
        model,
        input_data=[img, num, index],
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
        depth=4,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train setup
    model.train()  # Set the model to training mode
    model.cuda()  # Move the model to GPU if available

    loss_history = []  # To store loss values for each batch

    # Training loop
    epochs = 1
    tqdm.write("Starting training...")
    tqdm.write(f"Number of batches per epoch: {len(dataloader)}")
    tqdm.write(f"Total number of batches: {len(dataloader) * epochs}")
    tqdm.write(f"Batch size: {dataloader.batch_size}")
    with tqdm(total=len(dataloader) * epochs, desc="Training Progress") as pbar:
        dataset_percent = len(dataloader) // 50
        for epoch in range(epochs):
            for batch, (img, num, index, target) in enumerate(dataloader):
                img = img.cuda()  # Move to GPU if available
                num = num.cuda()  # Move to GPU if available
                index = index.cuda()  # Move to GPU if available
                target = target.cuda()  # Move to GPU if available

                optimizer.zero_grad()  # Zero the gradients
                output = model(img, num, index)  # Forward pass
                loss = loss_fn(output.squeeze(), target)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights

                pbar.update(1)  # Update the progress bar
                if batch % dataset_percent == 0 and batch != 0:
                    loss_history.append(loss.item())  # Store the loss value
                    tqdm.write(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Model saved to simple_cnn.pth")

    # Plot the loss history and save it
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig("loss_history.png")
    print("Loss history saved to loss_history.png")
    plt.close()

    # Evaluate the model on a single batch
    model.eval()
    with torch.no_grad():
        # Create a empty tensor to be concatenated (n * 12 * 28 * 28), where n is the batch size
        image_tensor = torch.full((10, 1, 28, 28), -1.0)
        num_tensor = torch.tensor([i / 9.0 for i in range(10)], dtype=torch.float32)

        image_tensor = image_tensor.cuda()
        num_tensor = num_tensor.cuda()

    with torch.no_grad():
        # Create a empty tensor to be concatenated (n * 12 * 28 * 28), where n is the batch size
        image_tensor = torch.full((10, 1, 28, 28), -1.0)
        num_tensor = torch.tensor([i / 9.0 for i in range(10)], dtype=torch.float32)

        image_tensor = image_tensor.cuda()
        num_tensor = num_tensor.cuda()

        for i in range(784):
            pixel_index = torch.full((10,), i / 784.0, dtype=torch.float32)
            pixel_index = pixel_index.cuda()

            output = model(image_tensor, num_tensor, pixel_index)
            image_tensor[:, 0].view(10, -1)[:, i] = output.squeeze()

        print(f"{image_tensor[:, 0]}")

        for i, image in enumerate(image_tensor[:, 0]):
            # Save the tensor as an image
            image = image.cpu().numpy()  # Move to CPU and convert to numpy
            plt.imshow(image, cmap="gray")
            plt.axis("off")  # Hide axes
            plt.savefig(f"output_image_{i}.png", bbox_inches="tight", pad_inches=0)
            plt.close()