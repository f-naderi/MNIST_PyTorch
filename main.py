# main.py
import torch
import argparse
from utils import get_data_loaders, get_device
from model import create_model
from train import train_model, save_model
from evaluate import (
    print_classification_report,
    plot_confusion_matrix,
    visualize_predictions,
    plot_training_history,
    plot_roc_curves,           
    plot_precision_recall_curves
)


def main():
    # Parse command-line arguments   
    parser = argparse.ArgumentParser(description='MNIST Training with PyTorch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='model/mnist_model.pth', help='Model save path')
    parser.add_argument('--no_train', action='store_true', help='Skip training')
    parser.add_argument('--load_model', type=str, default=None, help='Load pretrained model')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MNIST Classification with PyTorch")
    print("="*60)
    
    # 1. Set up the device 
    device = get_device()
    
    # 2. Load the data
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    
    # 3. Create or load the model   
    model = create_model(device)
    
    # Calculate the number of parameters  
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Load a pretrained model (if specified)
    if args.load_model:
        print(f"\nLoading pretrained model from {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    
    # 4. Train model 
    if not args.no_train:
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr
        )
        
          
        plot_training_history(history)
        
        save_model(model, args.save_path)
    
    # 5. Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    # Classification report
    print_classification_report(model, test_loader, device)
    
    # plot confusion matrix 
    plot_confusion_matrix(model, test_loader, device)
    
    # visualize predictions  
    visualize_predictions(model, test_loader, device, num_images=10)
    
    # 6. Plot ROC و Precision-Recall Diagrams
    plot_roc_curves(model, test_loader, device)
    plot_precision_recall_curves(model, test_loader, device)
    
    print("\nDone!")

if __name__ == "__main__":
    main()