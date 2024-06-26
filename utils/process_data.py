import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os
# import albumentations as A
import numpy as np

IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B = 0.485, 0.456, 0.406
IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B = 0.229, 0.224, 0.225

CLIP_MEAN_R, CLIP_MEAN_G, CLIP_MEAN_B = 0.481, 0.457, 0.408
CLIP_STD_R, CLIP_STD_G, CLIP_STD_B = 0.268, 0.261, 0.275

NORM_MEAN_R, NORM_MEAN_G, NORM_MEAN_B = 0.5, 0.5, 0.5
NORM_STD_R, NORM_STD_G, NORM_STD_B = 0.25, 0.25, 0.25


class SimulateSnow(object):
    def __init__(self, snow_coeff=0.2):
        self.snow_coeff = snow_coeff

    def __call__(self, img):
        snow_image = img + self.snow_coeff * torch.randn_like(img)
        return torch.clamp(snow_image, min=0, max=1)

class SimulateRain(object):
    def __init__(self, rain_coeff=0.2):
        self.rain_coeff = rain_coeff

    def __call__(self, img):
        rain_image = img + self.rain_coeff * torch.randn_like(img)
        return torch.clamp(rain_image, min=0, max=1)

class SimulateFog(object):
    def __init__(self, fog_coeff=0.2):
        self.fog_coeff = fog_coeff

    def __call__(self, img):
        fog_image = img + self.fog_coeff * torch.randn_like(img)
        return torch.clamp(fog_image, min=0, max=1)


def preprocess_mask(mask):
    ""
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
        transforms.ToTensor()
    ])
    mask = transform(mask)
    mask = mask * 255

    return mask


def preprocess_train(img):
    train_transform = transforms.Compose([
        transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
        transforms.ToTensor(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2)
        # transforms.Normalize(
        #     mean=[IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B], 
        #     std=[IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B]
        # )
    ])

    return train_transform(img)


def preprocess_train_robustness(image):
    # # Define a list to hold the transformations
    # train_transforms = []
    
    # # resize to 512x512
    # train_transforms.append(A.Resize(512, 512))

    # prob = torch.rand(1)

    # # Randomly select weather conditions based on their probabilities
    # if prob < 0.2:
    #     # RAIN
    #     train_transforms.append(A.RandomRain(p=0.7))
    #     # add fog
    #     if torch.rand(1) < 0.3:
    #         train_transforms.append(A.RandomFog())
    # elif prob >= 0.2 and prob < 0.4:
    #     # SNOW
    #     train_transforms.append(A.RandomSnow(snow_point_lower=0.3, snow_point_upper=0.5, brightness_coeff=2.5, p=0.7))
    #     # add fog
    #     # if torch.rand(1) < 0.3:
    #     #     train_transforms.append(A.RandomFog(p=0.7))
    # elif prob >= 0.4 and prob < 0.6:
    #     # JUST FOG
    #     train_transforms.append(A.RandomFog(p=0.7))
    # elif prob >= 0.6 and prob < 0.8:
    #     # SUN FLARE
    #     train_transforms.append(A.RandomSunFlare(p=0.7))
    # elif prob >= 0.8 and prob < 1.0:
    #     # SHADOW
    #     train_transforms.append(A.RandomShadow(p=0.7))
    
    # # Add random brightness and contrast changes
    # if torch.rand(1) < 0.5:
    #     train_transforms.append(A.RandomBrightnessContrast())
    
    # # Compose all transformations
    # transformation_piepline = A.Compose(train_transforms)
    
    # # transform image to numpy and apply transform:
    # transformed_image = transformation_piepline(image=np.array(image))['image']

    # # Apply the composed transformations to the image
    # return transforms.ToTensor()(transformed_image)

    # for prewrittne images
    tr = transforms.Compose([transforms.ToTensor()])

    return tr(image)



def preprocess(img):
    '''
    Function used at validation tim to resize, reshape, normalize image in theway the model expects it
    

    '''
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_R], 
        #     std=[IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B]
        # )
    ])
    img = transform(img)
    # img = img.unsqueeze(0)

    return img


def postprocess(prediction, shape):
    """
    Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation.
    
    """
    # # # softmax to get class for each pixel
    # # m = torch.nn.Softmax(dim=1)
    # # prediction_soft = m(prediction)

    # # get the class with the highest probability
    # prediction_max = torch.argmax(prediction, axis=1)
    # # resize to original image size
    # prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)
    # # convert shape: 4, 1024, 2048 (b, h, w) -> 1024, 2048, 4
    # prediction = prediction.permute(1, 2, 0)
    # # convert to numpy
    # prediction = prediction.cpu().detach().numpy()

    # return prediction
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy



def plot_images_with_masks(dataset, indices, num_images_per_row=2, title="default_title",
                           save=False, save_path="./results/plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(10, 5*num_rows))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        # img, mask = preprocess_train(img), preprocess_train(mask)
        ax = axes[i // num_images_per_row, i % num_images_per_row] if num_rows > 1 else axes[i % num_images_per_row]
        ax.imshow(img.permute(1, 2, 0))
        # ax.imshow(mask.permute(1, 2, 0), alpha=0.35, cmap='jet')
        ax.set_title(f'Image {idx}')
        ax.axis('off')

    plt.tight_layout()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{title}.png'))
    
    plt.show()


def plot_images_predictions_masks(images, predictions, masks, indices, num_images_per_row=2, title="default_title",
                                  save=False, save_path="./plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row * 2, figsize=(10, 5*num_rows))

    for i, idx in enumerate(indices):
        img = images[idx]
        pred = predictions[idx]
        mask = masks[idx]

        ax_img = axes[i // num_images_per_row, 2 * (i % num_images_per_row)]
        ax_pred = axes[i // num_images_per_row, 2 * (i % num_images_per_row) + 1]

        ax_img.imshow(img.cpu().permute(1,2,0).numpy())
        # ax_img.imshow(pred.argmax(dim=0).cpu().numpy(), alpha=0.35, cmap='jet')
        ax_img.set_title(f'Mask {idx}')
        ax_img.axis('off')

        ax_pred.imshow(pred.argmax(dim=0).cpu().numpy(), alpha=0.35, cmap='jet')
        ax_pred.set_title(f'Prediction {idx}')
        ax_pred.axis('off')

    plt.tight_layout()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{title}.png'))

    plt.show()


def plot_losses_and_dice(train_losses, val_losses, val_dices, 
                         save=True, save_path="./plots/losses", model_name="deeplabv3plus_ce.pt"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, val_dices, label='Validation Dice Score')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Dice Score')
    plt.legend()
    plt.grid(True)

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Get filename and replace extension
        root, ext = os.path.splitext(model_name)
        filename = root + ".png"
        # Save the plot with the new filename
        plt.savefig(os.path.join(save_path, filename))

    plt.show()