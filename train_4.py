import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchinfo
from tqdm import tqdm


# Create the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.norm4 = nn.BatchNorm2d(256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.norm1(self.gelu(self.conv1(x)))
        x = self.norm2(self.gelu(self.conv2(x)))
        x = self.max_pool(x)
        x = self.norm3(self.gelu(self.conv3(x)))
        x = self.norm4(self.gelu(self.conv4(x)))
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.output(x)
        x = self.sigmoid(x)

        return x


class MyMNISTTrainDataset(Dataset):
    def __init__(self):
        self.dataset = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, num = self.dataset[idx]

        # Get a random number between 0 and 784 as the index
        index = torch.randint(0, 784, (1,)).item()

        # Get the pixel at the specified index as the target label
        target_pixel = img_tensor.view(-1)[index].detach().clone()

        # Create a mask tensor with 784 elements
        mask = torch.arange(784) >= index
        # Apply the mask to the image tensor
        img_tensor.view(-1)[mask] = -1.0

        # Create a new tensor to encode the position
        pos_encoding_tensor = torch.full_like(img_tensor, index / 784.0)

        # Create a new tensor to encode the target number
        num_encoding_tensor = torch.full_like(img_tensor, num / 9.0)

        # Concatenate the number encoding tensor with the image tensor
        img_tensor = torch.cat((img_tensor, pos_encoding_tensor, num_encoding_tensor), dim=0)

        return img_tensor, target_pixel


if __name__ == "__main__":
    dataset = MyMNISTTrainDataset()

    data, target = dataset[0]
    print(f"Data shape: {data.shape}, Target: {target}")

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
        num_workers=12,
        persistent_workers=True,
    )

    # # Print first batch to test the dataset
    # for data, target in dataloader:
    #     print(f"Data: {data}, Target: {target}")
    #     break

    # Get type of the first batch
    for data, target in dataloader:
        print(f"Data type: {data.dtype}, Target type: {target.dtype}")
        break

    model = SimpleCNN()

    torchinfo.summary(
        model,
        input_size=(32, 3, 28, 28),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train setup
    model.train()  # Set the model to training mode
    model.cuda()  # Move the model to GPU if available

    loss_history = []  # To store loss values for each batch

    # Training loop
    epochs = 1000
    tqdm.write("Starting training...")
    tqdm.write(f"Number of batches per epoch: {len(dataloader)}")
    tqdm.write(f"Total number of batches: {len(dataloader) * epochs}")
    tqdm.write(f"Batch size: {dataloader.batch_size}")
    with tqdm(total=len(dataloader) * epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            for batch, (data, target) in enumerate(dataloader):
                data, target = data.cuda(), target.cuda()  # Move data to GPU if available

                optimizer.zero_grad()  # Zero the gradients
                output = model(data)  # Forward pass
                loss = loss_fn(output.squeeze(), target)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights

                pbar.update(1)  # Update the progress bar
            tqdm.write(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")
            loss_history.append(loss.item())  # Store the loss value

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
        # Create a empty tensor to be concatenated (n * 3 * 28 * 28), where n is the batch size
        batched_input = torch.empty((0, 3, 28, 28))

        for i in range(10):
            # Create an 28*28 tensor with -1.0 values
            masked_image = torch.full((1, 28, 28), -1.0)

            # Create a position encoding tensor
            hot_pos_tensor = torch.full((1, 28, 28), 0.0)

            # Concatenate the hot position tensor with the masked image
            input_tensor = torch.cat((masked_image, hot_pos_tensor), dim=0)

            # Create a number encoding tensor
            num_encoding_tensor = torch.full((1, 28, 28), i / 9.0)

            # Concatenate the number encoding tensor with the input tensor
            input_tensor = torch.cat((input_tensor, num_encoding_tensor), dim=0)

            # Add the input tensor to the batch
            batched_input = torch.cat((batched_input, input_tensor.unsqueeze(0)), dim=0)

        batched_input = batched_input.cuda()
        
        for i in range(0, 784):
            batched_input[:, 1] = i / 784.0  # Set the position encoding tensor to the current index normalized by 784

            output = model(batched_input)  # Forward pass
            batched_input[:, 0].view(10, -1)[:, i] = output.squeeze()  # Update the first pixel in the batch
            

        print(f"{batched_input[:, 0]}")

        for i, image in enumerate(batched_input[:, 0]):
            # Save the tensor as an image
            image = image.cpu().numpy()  # Move to CPU and convert to numpy
            plt.imshow(image, cmap='gray')
            plt.axis('off')  # Hide axes
            plt.savefig(f"output_image_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()