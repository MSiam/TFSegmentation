from PIL import Image
import numpy as np

# colour map
label_colours_global = [(128, 64, 128),  # 'road'
                        (244, 35, 232),  # 'sidewalk'
                        (70, 70, 70),  # 'building'
                        (102, 102, 156),  # 'wall'
                        (190, 153, 153),  # 'fence'
                        (153, 153, 153),  # 'pole'
                        (250, 170, 30),  # 'traffic light'
                        (220, 220, 0),  # 'traffic sign'
                        (107, 142, 35),  # 'vegetation'
                        (152, 251, 152),  # 'terrain'
                        (70, 130, 180),  # 'sky'
                        (220, 20, 60),  # 'person'
                        (255, 0, 0),  # 'rider'
                        (0, 0, 142),  # 'car'
                        (0, 0, 70),  # 'truck'
                        (0, 60, 100),  # 'bus'
                        (0, 80, 100),  # 'train'
                        (0, 0, 230),  # 'motorcycle'
                        (119, 11, 32),  # 'bicycle'
                        (0, 0, 0), ]  # None
#label_colours_global = [(0,0,0),
#                        (0,0,0),
#                        (128, 64, 128),  # 'road'
#                        (244, 35, 232),  # 'sidewalk'
#                        (70, 70, 70),  # 'building'
#                        (102, 102, 156),  # 'wall'
#                        (190, 153, 153),  # 'fence'
#                        (153, 153, 153),  # 'pole'
#                        (250, 170, 30),  # 'traffic light'
#                        (220, 220, 0),  # 'traffic sign'
#                        (107, 142, 35),  # 'vegetation'
#                        (152, 251, 152),  # 'terrain'
#                        (70, 130, 180),  # 'sky'
#                        (220, 20, 60),  # 'person'
#                        (255, 0, 0),  # 'rider'
#                        (0, 0, 142),  # 'car'
#                        (0, 0, 70),  # 'truck'
#                        (0, 60, 100),  # 'bus'
#                        (0, 80, 100),  # 'train'
#                        (0, 0, 230),  # 'motorcycle'
#                        (119, 11, 32),  # 'bicycle'
#                        ]  # None



def decode_labels(mask, num_classes):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking **argmax**.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
#    num_classes= num_classes+1
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
    assert (num_classes == len(colours)), 'num_classes %d should be equal the number colours %d.' % (num_classes, len(colours))
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
