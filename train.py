"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# from model import Model
from custom_model import DeepLabV3Plus
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser

from utils.process_data import plot_images_with_masks, preprocess, preprocess_train, preprocess_mask, plot_images_predictions_masks
from utils.train_test_utils import train, dice_score
from utils.utils import map_id_to_train_id

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


def ex_main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # define batch size and epochs
    batch_size = 32
    epochs = 15
    k_folds = 5

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
    
    # USE THIS WITH KFOLD
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    # USE THIS WHEN MOVING AWAY FROM KFOLD
    # # start from lr = (1 - curr_iter/max_iter)^0.9
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    # # step_size = 1 -> decay weight every epoch
    # # gamma = 0.1 -> deccay factor
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

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
        val_dices = []
        for epoch in range(1, epochs + 1):
            # shuffle training indices each epoch if desired
            random.shuffle(train_indices)

            # train
            model.train()
            total_train_loss = 0.0
            num_train_batches = len(train_indices) // batch_size
            for i in range(0, len(train_indices), batch_size):
                # create a batch
                batch_indices = train_indices[i:i+batch_size]
                images, masks = [], []
                for idx in batch_indices:
                    img, mask = dataset[idx]
                    # preproc
                    images.append(preprocess(img))
                    masks.append(preprocess_mask(mask))
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

                total_train_loss += loss.item()
            train_losses.append(total_train_loss / num_train_batches)
            print(f"Epoch {epoch}: Training Loss: {train_losses[-1]:.4f}")

            # validation
            model.eval()
            total_val_dice = 0.0
            total_val_loss = 0.0
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
                    loss = criterion(outputs, masks.squeeze().long())

                    total_val_dice += dice.item()
                    total_val_loss += loss.item()
                val_losses.append(total_train_loss / num_val_batches)
                val_dices.append(total_val_dice / num_val_batches)
                print(f"Epoch {epoch}: Validation Loss: {val_losses[-1]:.4f}")
                print(f"Epoch {epoch}: Validation Dice Score: {val_dices[-1]:.4f}")

    # save model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_ce.pth"))


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    # set cuda reservation max split size
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' 

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # define batch size and epochs
    batch_size = 16
    epochs = 50

    # data loading
    trainset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=preprocess_train, target_transform=preprocess_mask)
    valset = Cityscapes(args.test_data_path, split='val', mode='fine', target_type='semantic',
                        transform=preprocess, target_transform=preprocess_mask)

    # visualize example images
    # plot_images_with_masks(dataset, indices=[0, 1, 2, 3, 4, 5, 6, 7], num_images_per_row=4,
    #                        save=False, save_path="./results/plots"    )

    # save original image height and width
    # img_w, img_h = trainset[0][0].size

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


    # Define the model and optimizer
    model = DeepLabV3Plus(num_classes=34).to(device)

    # start from lr = (1 - curr_iter/max_iter)^0.9
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
    # step_size = 1 -> decay weight every epoch
    # gamma = 0.1 -> deccay factor
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # patience for early stopping
    patience = 5
    patience_threshold = 0.03
    saved_by_early_stopping = False

    # training
    train_losses = []
    val_losses = []
    val_dices = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")

        # train
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader)
        for (images, masks) in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            # images shape: b, 3, 512, 512
            outputs = model(images) 
            # outputs shape: b, 34, 512, 512 -> 1024, 2048, b -> no postproc yet???? also mask resized
            # outputs = postprocess(outputs, (img_h, img_w))
            loss = criterion(outputs, masks.squeeze().long())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # clear cuda memory
            gc.collect()
            torch.cuda.empty_cache()
        train_losses.append(total_train_loss / num_train_batches)
        print(f"Training Loss: {train_losses[-1]:.4f}")

        # validation
        model.eval()
        total_val_dice = 0.0
        total_val_loss = 0.0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for (images, masks) in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                dice = dice_score(outputs, masks.squeeze())
                loss = criterion(outputs, masks.squeeze().long())
                total_val_dice += dice.item()
                total_val_loss += loss.item()
            val_losses.append(total_val_loss / num_val_batches)
            val_dices.append(total_val_dice / num_val_batches)

            # clear cuda memory
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Validation Loss: {val_losses[-1]:.4f}")
        print(f"Validation Dice Score: {val_dices[-1]:.4f}")
        
        scheduler.step()

        # periodic model saving
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f"deeplabv3plus_ce_e{10}.pth"))

        # early stopping if validation stops decreasing
        if epoch > 10 \ 
            and abs(total_val_loss - (sum(prev_val_losses[-(patience - 1):]) / patience)) < patience_threshold:
            
            print("Stopping Early...")
            # save model
            torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_ce.pth"))
            saved_by_early_stopping = True

    if not saved_by_early_stopping:
        # save model
        torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_ce.pth"))


def test_model(args, model_name="deeplabv3plus_ce.pth"):
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model from .pth file
    model = DeepLabV3Plus(num_classes=34).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, model_name), 
                                    map_location=torch.device(device)))

    # define batch size
    batch_size = 16

    # load_data
    testset = Cityscapes(args.test_data_path, split='test', mode='fine', target_type='semantic',
                        transform=preprocess, target_transform=preprocess_mask)
    
    testset.images = testset.images[:8]
    testset.targets = testset.targets[:8]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Testing
    model.eval()
    total_test_dice = 0.0
    totalt_test_loss = 0.0
    num_test_batches = len(test_loader)
    with torch.no_grad():
        curr_batch = 0
        for (images, masks) in test_loader:
            images = images.to(device)
            masks = map_id_to_train_id(masks).to(device)

            outputs = model(images)
            dice = dice_score(outputs, masks.squeeze())
            loss = criterion(outputs, masks.squeeze().long())
            total_test_dice += dice.item()
            totalt_test_loss += loss.item()

            if random.random() < 0.2:
                # generate 4 random indices in range batch_size
                indices = random.sample(range(batch_size), 4)
                plot_images_predictions_masks(images, outputs, masks, 
                                        indices=range(batch_size), num_images_per_row=2, 
                                        title=f"Test Batch {curr_batch}", save=True)
            
            curr_batch += 1
            
            # clear cuda memory
            gc.collect()
            torch.cuda.empty_cache()

    test_dice_score = total_test_dice / num_test_batches
    test_loss = totalt_test_loss / num_test_batches
    print(f"Testing Loss: {test_loss:.4f}")
    print(f"Testing Dice Score: {test_dice_score:.4f}")


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
    test_model(args, model_name="deeplabv3plus_ce.pth")
