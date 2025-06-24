from train_8 import SimpleCNN
import torch
import torchinfo
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the trained model from disk
    model = SimpleCNN()
    model.load_state_dict(torch.load("simple_cnn.pth"))
    model.eval()
    model.cuda()  # Move the model to GPU if available
    print("Model loaded")

    # # Print the model summary
    # torchinfo.summary(
    #     model,
    #     input_size=(32, 1, 28, 28),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     depth=4,
    # )

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
