"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# from model import Model
from custom_model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser

from utils.process_data import *
from utils.train_test_utils import train, dice_score
from utils.utils import map_id_to_train_id
from utils.poly_scheduler import PolyLR
from utils.denormalize import denormalize

from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import random
import os
import gc

from torchmetrics.classification import Dice


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    parser.add_argument("--test_data_path", type=str, default=".", help="Path to the test data")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path where .pth files are saved")
    parser.add_argument("--plot_save_path", type=str, default="./plots", help="Path where plots are saved")
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    # set cuda reservation max split size
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' 

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # data loading
    # TRAIN = TRAIN_ROBUST when using args.test_data_path
    trainset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=preprocess_train, target_transform=preprocess_mask)
    valset = Cityscapes(args.test_data_path, split='val', mode='fine', target_type='semantic',
                        transform=preprocess, target_transform=preprocess_mask)

    # visualize example images
    # plot_images_with_masks(dataset, indices=[0, 1, 2, 3, 4, 5, 6, 7], num_images_per_row=4,
                        #    save=False, save_path="./results/plots"    )

    # save original image height and width
    # img_w, img_h = trainset[0][0].size

    # define batch size and epochs
    batch_size = 16

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Define the model and optimizer
    model = Model().to(device)

    epochs = 50
    # sgd
    lr = 0.1
    # adam
    # lr = 0.01

    # start from lr = (1 - curr_iter/max_iter)^0.9
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=1e-4)
    # step_size = 1 -> decay weight every epoch
    # gamma = 0.1 -> deccay factor
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = PolyLR(optimizer, epochs, power=0.9)
    # dont forget scheduler

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # loss
    # changed utils.py: 255 -> 33
    criterion = torch.nn.CrossEntropyLoss(ignore_index=33)

    # scorer
    scorer = Dice(num_classes=34, ignore_index=33).to(device)

    # patience for early stopping
    patience = 5
    patience_threshold = 0.02
    stopped_early = False

    # training
    train_losses = []
    val_losses = []
    val_dices = []
    for epoch in range(1, epochs + 1):
        if stopped_early:
            break

        print(f"\nEpoch {epoch}")

        # train
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader)
        for (images, masks) in train_loader:
            images = images.to(device)
            masks = map_id_to_train_id(masks).to(device)

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
                images = images.to(device)
                masks = map_id_to_train_id(masks).to(device)

                outputs = model(images)
                # dice = dice_score(outputs, masks)
                dice = scorer(outputs.argmax(dim=1), masks.squeeze().long())
                loss = criterion(outputs, masks.squeeze().long())
                # dice.mean() for mean across channels (classes)
                # dice.sum() for sum across channels (classes)
                total_val_dice += dice.mean().item()
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
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f"final_deeplab_sgd_ce_e{epoch}_none_lr{lr}.pth"))

        # # early stopping if validation stops decreasing
        # loss_difference = abs(val_losses[-1] - (sum(val_losses[-(patience - 1):]) / (patience - 1)))
        # print(loss_difference)
        # if epoch > patience and loss_difference < patience_threshold:
        #     print("Stopping Early...")
        #     # save model
        #     torch.save(model.state_dict(), os.path.join(args.model_save_path, "deeplabv3plus_adam_ce.pth"))
        #     stopped_early = True

    if not stopped_early:
        # save model
        torch.save(model.state_dict(), os.path.join(args.model_save_path, f"final_deeplab_sgd_ce_e{epochs}_none_lr{lr}.pth"))

    # plot losses and dice
    plot_losses_and_dice(train_losses, val_losses, val_dices, 
                         save=True, save_path="./plots/losses", 
                         model_name=f"final_deeplab_sgd_ce_e{epochs}_robust_lr{lr}.pth")


def test_model(args, model_name="deeplabv3plus_ce.pth", norm=False):
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model from .pth file
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, model_name), 
                                    map_location=torch.device(device)))

    # define batch size
    batch_size = 16

    # load_data
    testset = Cityscapes(args.test_data_path, split='test', mode='fine', target_type='semantic',
                        transform=preprocess, target_transform=preprocess_mask)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # loss
    # changed utils.py: 255 -> 33
    criterion = torch.nn.CrossEntropyLoss(ignore_index=33)

    # dice score
    scorer = Dice(num_classes=34, ignore_index=33).to(device)

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
            # print("OUT:", outputs.shape, masks.shape, masks.squeeze().shape)
            # dice = dice_score(outputs, masks)
            dice = scorer(outputs.argmax(dim=1), masks.squeeze().long())
            loss = criterion(outputs, masks.squeeze().long())
            # dice.mean() for mean across channels (classes)
            # dice.sum() for sum across channels (classes)
            total_test_dice += dice.mean().item()
            totalt_test_loss += loss.item()

            if random.random() < 0.2:
                # generate 4 random indices in range batch_size
                indices = random.sample(range(batch_size), 4)
                if norm:
                    images = denormalize(
                        images,
                        mean=[IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B],
                        std=[IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B]
                    )
                    # images = images * 255
                plot_images_predictions_masks(images, outputs, masks, 
                                        indices=range(batch_size), num_images_per_row=2, 
                                        title=f"Test Batch {curr_batch}", 
                                        save=True, save_path=f"./plots/{model_name}")
            
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
    test_model(args, model_name="final_deeplab_sgd_ce_e40_none_lr0.1.pth", norm=False)
