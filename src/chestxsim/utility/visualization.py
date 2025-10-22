import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['image.cmap'] = 'gray'
_original_imshow = plt.imshow

def imshow_with_colorbar(*args, **kwargs):
    img = _original_imshow(*args, **kwargs)

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    return img

plt.imshow = imshow_with_colorbar

def plot(*images):
    n_images = len(images)
    if n_images == 1:
        plt.figure(figsize=(5, 4))
        plt.imshow(images[0])
    else:

        if n_images <= 3:
            ncols = n_images
            nrows = 1
        elif n_images <= 6:
            ncols = 3
            nrows = 2
        elif n_images <= 9:
            ncols = 3
            nrows = 3
        else:
            ncols = 4
            nrows = (n_images + ncols - 1) // ncols
        
        # Auto-size figure based on grid
        plt.figure(figsize=(4 * ncols, 3.5 * nrows))
        
        for i, img in enumerate(images, 1):
            plt.subplot(nrows, ncols, i)
            plt.imshow(img)
              
       
    plt.tight_layout()
    plt.show()