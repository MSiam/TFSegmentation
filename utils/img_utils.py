from PIL import Image
import numpy as np

# colour map
label_colours_global = [(0, 0, 0),
                        (128, 0, 0),
                        (0, 128, 0),
                        (128, 128, 0),
                        (0, 0, 128),
                        (128, 0, 128),
                        (0, 128, 128),
                        (128, 128, 128),
                        (64, 0, 0),
                        (192, 0, 0),
                        (64, 128, 0),
                        (192, 128, 0),
                        (64, 0, 128),
                        (192, 0, 128),
                        (64, 128, 128),
                        (192, 128, 128),
                        (0, 64, 0),
                        (128, 64, 0),
                        (0, 192, 0),
                        (128, 192, 0),
                        (0, 64, 128)]


def decode_labels(mask, num_classes):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking **argmax**.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """

    # init colours array
    colours = label_colours_global

    # if num_classes == 7:
    #     colours = label_colours_scala_7
    # elif num_classes == 6:
    #     colours = label_colours_scala_6
    # elif num_classes == 5:
    #     colours = label_colours_scala_5
    # else:
    #     print("ERROR this number of classes don't have a defined colours")
    #     exit(-1)

    # Check the length of the colours with num_classes
    assert (num_classes <= len(colours)), 'num_classes %d should be less or equal than number colours %d.' % (num_classes, len(colours))
    # Get the shape of the mask
    n, h, w = mask.shape
    # Create the output numpy array
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    # Loop on images
    for i in range(n):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = colours[k]
        outputs[i] = np.array(img)
    return outputs
