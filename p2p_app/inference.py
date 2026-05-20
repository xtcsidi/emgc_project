import torch
from torchvision import datasets, transforms
import random
import argparse
import sys
from node import MNISTCNN

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    RICH = True
except ImportError:
    RICH = False

def print_ascii_digit(image_tensor):
    """Prints a 28x28 tensor as ascii art in the terminal"""
    pixels = image_tensor.squeeze().numpy()
    chars = " .:-=+*#%@"
    
    art = ""
    for row in pixels:
        line = ""
        for p in row:
            # Denormalize to 0-1 for ascii mapping
            val = (p * 0.3081) + 0.1307
            val = max(0, min(1, val))
            idx = int(val * (len(chars) - 1))
            line += chars[idx] * 2  # *2 makes it look more square in terminal
        art += line + "\n"
    
    if RICH:
        console.print(Panel(art, title="[cyan]Input Image (Handwritten Digit)[/cyan]", expand=False))
    else:
        print("\n--- Input Image ---")
        print(art)
        print("-------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Test your federated trained model!")
    parser.add_argument("--model", type=str, required=True, help="Path to your saved .pth model file")
    args = parser.parse_args()

    if RICH:
        console.print(f"\n[bold yellow]Loading trained model weights from:[/bold yellow] {args.model}")
    else:
        print(f"\nLoading trained model weights from: {args.model}")

    # 1. Load the model architecture
    model = MNISTCNN()
    
    # 2. Load your trained weights
    try:
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.eval()

    # 3. Get the MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Suppress download logs if it needs to download
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Pick a random image from the test set
    idx = random.randint(0, len(test_dataset) - 1)
    image, true_label = test_dataset[idx]

    # Show the image to the user as ASCII art
    print_ascii_digit(image)

    # 4. Use the model to predict what digit this is
    with torch.no_grad():
        # Add batch dimension: [1, 1, 28, 28]
        output = model(image.unsqueeze(0))
        
        # Get the index of the highest probability
        prediction = output.argmax(dim=1).item()

    if RICH:
        console.print(f"[bold cyan]True Label:[/bold cyan] {true_label}")
        console.print(f"[bold magenta]Model Prediction:[/bold magenta] {prediction}")
        if prediction == true_label:
            console.print("\n[bold green][SUCCESS] Your Federated Model recognized the digit perfectly![/bold green]\n")
        else:
            console.print("\n[bold red][INCORRECT] The model made a mistake on this one.[/bold red]\n")
    else:
        print(f"True Label: {true_label}")
        print(f"Model Prediction: {prediction}")
        if prediction == true_label:
            print("\n[SUCCESS] Your Federated Model recognized the digit perfectly!\n")
        else:
            print("\n[INCORRECT] The model made a mistake on this one.\n")

if __name__ == "__main__":
    main()
