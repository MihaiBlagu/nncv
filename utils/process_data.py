import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os


IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B = 0.485, 0.456, 0.406
IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B = 0.229, 0.224, 0.225

CLIP_MEAN_R, CLIP_MEAN_G, CLIP_MEAN_B = 0.481, 0.457, 0.408
CLIP_STD_R, CLIP_STD_G, CLIP_STD_B = 0.268, 0.261, 0.275

NORM_MEAN_R, NORM_MEAN_G, NORM_MEAN_B = 0.5, 0.5, 0.5
NORM_STD_R, NORM_STD_G, NORM_STD_B = 0.25, 0.25, 0.25


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
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B], 
        #     std=[IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B]
        # )
    ])

    return train_transform(img)


def preprocess(img):
    '''
    Function used at validation tim to resize, reshape, normalize image in theway the model expects it
    

    '''
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_R], 
        #                      std=[IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B])
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
    # softmax to get class for each pixel
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    # get the class with the highest probability
    prediction_max = torch.argmax(prediction_soft, axis=1)
    # resize to original image size
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)
    # convert shape: 4, 1024, 2048 (b, h, w) -> 1024, 2048, 4
    prediction = prediction.permute(1, 2, 0)
    # convert to numpy
    prediction = prediction.cpu().detach().numpy()

    return prediction


def plot_images_with_masks(dataset, indices, num_images_per_row=2, title="default_title",
                           save=False, save_path="./results/plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(10, 5*num_rows))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img, mask = preprocess(img), preprocess(mask)
        ax = axes[i // num_images_per_row, i % num_images_per_row] if num_rows > 1 else axes[i % num_images_per_row]
        ax.imshow(img.permute(1, 2, 0))
        ax.imshow(mask.permute(1, 2, 0), alpha=0.35, cmap='jet')
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
        ax_img.imshow(pred.argmax(dim=0).cpu().numpy(), alpha=0.35, cmap='jet')
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