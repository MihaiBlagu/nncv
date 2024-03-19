"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# from model import Model
from custom_model import DeepLabV3Plus
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser

from utils.process_data import plot_images_with_masks, preprocess, postprocess, preprocess_mask
from utils.train_test_utils import train, dice_score

from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import random
import os


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path where .pth files are saved")
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # define batch size and epochs
    batch_size = 32
    epochs = 10
    k_folds = 3

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic')

    # visualize example images
    # plot_images_with_masks(dataset, indices=[0, 1, 2, 3, 4, 5, 6, 7], num_images_per_row=4,
    #                        save=False, save_path="./results/plots"    )
    

    # save original image height and width
    img_w, img_h = dataset[0][0].size

    # Initialize the k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Define the model and optimizer
    # model = Model().to(device)
    model = DeepLabV3Plus(num_classes=34).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    outputs_per_fold = []

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # define indices for current fold
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()

        # training
        train_losses = []
        for epoch in range(1, epochs):  # Assuming 100 epochs
            # shuffle training indices each epoch if desired
            random.shuffle(train_indices)

            total_loss = 0.0
            num_batches = len(train_indices) // batch_size
            for i in range(0, len(train_indices), batch_size):
                # create a batch
                batch_indices = train_indices[i:i+batch_size]
                images, masks = [], []
                for idx in batch_indices:
                    img, mask = dataset[idx]
                    # preproc
                    img = preprocess(img)
                    mask = preprocess_mask(mask)
                    images.append(img)
                    masks.append(mask)
                images = torch.stack(images).to(device)
                masks = torch.stack(masks).to(device)

                optimizer.zero_grad()
                # images shape: b, 3, 512, 512
                outputs = model(images) 
                # outputs shape: b, 34, 512, 512 -> 1024, 2048, b -> no postproc yet???? also mask resized
                # outputs = postprocess(outputs, (img_h, img_w))
                loss = criterion(outputs, masks.squeeze().long())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_losses.append(total_loss / num_batches)
            print(f"Epoch {epoch}: Training Loss: {train_losses[-1]:.4f}")

        # validation
        model.eval()
        total_val_dice = 0.0
        num_val_batches = len(val_indices) // batch_size
        for i in range(0, len(val_indices), batch_size):
            # create batches
            batch_indices = val_indices[i:i+batch_size]
            images, masks = [], []
            for idx in batch_indices:
                img, mask = dataset[idx]
                img = preprocess(img)  # Add batch dimension
                mask = preprocess_mask(mask)  # Add batch dimension
                images.append(img)
                masks.append(mask)
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).to(device)
            
            outputs = model(images)
            # outputs = postprocess(outputs, (img_h, img_w))
            dice = dice_score(outputs, masks.squeeze())
            total_val_dice += dice.item()

        val_dice = total_val_dice / num_val_batches
        print(f"Epoch {epoch}: Validation Dice Score: {val_dice:.4f}")

    # save model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_ce.pth"))

    # Visualize some results



if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
