"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# from model import Model
from custom_model import DeepLabV3Plus
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser

from utils.process_data import plot_images_with_masks, preprocess, postprocess, preprocess_mask, plot_images_predictions_masks
from utils.train_test_utils import train, dice_score

from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import random
import os

import gc



def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--test_data_path", type=str, default=".", help="Path to the test data")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path where .pth files are saved")
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # set cuda allocation size to 128MB to prevent gpu memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # set torch acllocation size
    os.environ

    # define batch size and epochs
    batch_size = 16
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

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # define indices for current fold
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()

        # training
        train_losses = []
        val_losses = []
        for epoch in range(1, epochs + 1):
            # shuffle training indices each epoch if desired
            random.shuffle(train_indices)

            # train
            model.train()
            total_loss = 0.0
            num_train_batches = len(train_indices) // batch_size
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
            train_losses.append(total_loss / num_train_batches)
            print(f"Epoch {epoch}: Training Loss: {train_losses[-1]:.4f}")

            # validation
            model.eval()
            total_val_dice = 0.0
            num_val_batches = len(val_indices) // batch_size
            with torch.no_grad():
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
                    dice = dice_score(outputs, masks.squeeze())
                    total_val_dice += dice.item()
                val_losses.append(total_val_dice / num_val_batches)
                print(f"Epoch {epoch}: Validation Dice Score: {val_losses[-1]:.4f}")

    # save model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_ce.pth"))

    # clean gpu memory after training
    gc.collect()
    torch.cuda.empty_cache()
    

def test_model(args, model_name="deeplabv3plus_ce.pth"):
    
    # set cuda allocation size to 128MB to prevent gpu memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model from .pth file
    model = DeepLabV3Plus(num_classes=34).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, model_name)))

    # define batch size
    batch_size = 4

    # visualize some reslults
    testset = Cityscapes(args.test_data_path, split='test', mode='fine', target_type='semantic')

    # Testing
    model.eval()
    total_test_dice = 0.0
    num_test_batches = len(testset) // batch_size
    # maybe these things occupy mem
    # test_outputs = []
    # test_masks = []
    # test_images = []
    with torch.no_grad():
        for i in range(0, len(testset), batch_size):
            # Create batches
            images, masks = [], []
            for idx in range(i, min(i + batch_size, len(testset))):
                img, mask = testset[idx]
                img = preprocess(img)  # Add batch dimension
                mask = preprocess_mask(mask)  # Add batch dimension
                images.append(img)
                masks.append(mask)
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).to(device)

            outputs = model(images)
            dice = dice_score(outputs, masks.squeeze())
            total_test_dice += dice.item()

            # test_outputs.append(outputs)
            # test_masks.append(masks)
            # test_images.append(images)
            
            # save batch 3
            if i == 3 or i == 9 or i == 12:
                plot_images_predictions_masks(images[i], outputs[i], masks[i], 
                                    indices=range(len(batch_size)), num_images_per_row=2, 
                                    title=f"Test Batch {i+1}", save=True)


        test_dice_score = total_test_dice / num_test_batches
        print(f"Testing Dice Score: {test_dice_score:.4f}")


    # # Randomly sample two batches
    # sampled_indices = random.sample(range(len(test_outputs)), 4)
    # sampled_images = [test_images[idx] for idx in sampled_indices]
    # sampled_masks = [test_masks[idx] for idx in sampled_indices]
    # sampled_outputs = [test_outputs[idx] for idx in sampled_indices]

    # # Plot the sampled batches
    # for i in range(2):  # Assuming you want to plot two batches
    #     plot_images_predictions_masks(sampled_images[i], sampled_outputs[i], sampled_masks[i], 
    #                                 indices=range(len(sampled_images[i])), num_images_per_row=2, 
    #                                 title=f"Test Batch {i+1}", save=False)

    # clean gpu memory after testing
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    # main(args)
    test_model(args, model_name="deeplabv3plus_ce.pth")
