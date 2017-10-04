import cv2
import skimage.io
import numpy as np
import skimage.transform


def make_sandwich(content_image, style_image, initial_generated_image=1, mask=False):
    img_height_width = gimme_img_size(content_image, True, pix_area_limit=100000)

    content_image = skimage.transform.resize(content_image, img_height_width, mode='reflect')
    content_image = np.expand_dims(content_image, 0)
    img_shape = content_image.shape

    style_image = skimage.transform.resize(style_image, img_height_width, mode='reflect')
    style_image = np.expand_dims(np.asarray(style_image, dtype=np.float32), 0)

    try:
        assert mask.dtype == 'float32' or mask.dtype == 'uint8'
        use_mask = True
        mask = skimage.transform.resize(mask, img_height_width, mode='reflect').astype(np.float32)
    except:
        use_mask = False
        mask = np.ones_like(content_image[0])

    if initial_generated_image == 1:
        # case_1 initial_generated_image = content_image.
        generated_image = content_image.copy()

    elif initial_generated_image == 2:
        # case_2 initial_generated_image = content_image + noise.
        generated_image = content_image.copy()
        add_noise = mask * np.random.uniform(0., 1., img_shape)
        generated_image = np.clip(generated_image + add_noise, 0, 1)

    elif initial_generated_image == 3:
        # case_3 initial_generated_image = noise.
        if use_mask:
            generated_image = invert_mask(mask) * content_image \
                              + mask[np.newaxis, ...] * \
                                np.random.uniform(0., 1., [1, img_shape[1], img_shape[2], img_shape[3]])
        else:
            generated_image = np.random.uniform(0., 1., [1, img_shape[1], img_shape[2], img_shape[3]])

    elif initial_generated_image == 4:
        # case_4 initial_generated_image = content_image with lumina transfer from style_image
        generated_image = content_image.copy()
        generated_image[0] = preserve_content_color(style_image[0].astype(np.float32),
                                                    generated_image[0].astype(np.float32))

    elif initial_generated_image == 5:
        # case_5 initial_generated_image = content_image with flipped (lumina transfer from style_image)
        generated_image = content_image.copy()
        generated_image[0] = preserve_content_color(
            flip_image(style_image[0].astype(np.float32)), generated_image[0].astype(np.float32))

    elif initial_generated_image == 6:
        # case_6 initial_generated_image = content_image + noise with flipped (lumina transfer from style_image)
        generated_image = content_image.copy()
        add_noise = mask * np.random.uniform(0., 1., img_shape)
        generated_image = np.clip(generated_image + add_noise, 0, 1)
        generated_image[0] = preserve_content_color(
            flip_image(style_image[0].astype(np.float32)), generated_image[0].astype(np.float32))

    elif initial_generated_image == 7:
        # case_7 initial_generated_image = noise with lumina transfer from style_image
        generated_image = mask[np.newaxis, ...] * np.random.uniform(0., 1.,
                                                                    [1, img_shape[1], img_shape[2], img_shape[3]])
        generated_image[0] = preserve_content_color(style_image[0].astype(np.float32),
                                                    generated_image[0].astype(np.float32))

    elif initial_generated_image == 8:
        # case_8 initial_generated_image = noise with flip-flipped (lumina transfer from style_image)
        generated_image = mask[np.newaxis, ...] * np.random.uniform(0., 1.,
                                                                    [1, img_shape[1], img_shape[2], img_shape[3]])
        generated_image[0] = preserve_content_color(flip_image(
            flip_image(style_image[0].astype(np.float32)), True), generated_image[0].astype(np.float32))

    if use_mask:
        return np.asarray(np.concatenate(
            (content_image, style_image, generated_image), axis=0), dtype=np.float32), mask
    else:
        return np.asarray(np.concatenate(
            (content_image, style_image, generated_image), axis=0), dtype=np.float32)


def invert_mask(mask):
    if mask.dtype == 'float32':
        invert_mask = np.ones_like(mask)
        return np.abs(mask - invert_mask)
    elif mask.dtype == 'uint8':
        invert_mask = np.ones_like(mask)
        return (255 * np.abs(mask/255 - invert_mask)).astype(np.uint8)


    invert_mask = np.ones_like(mask)
    return np.abs(mask - invert_mask)


def gimme_img_size(image, keep_original_proportion=True, pix_area_limit=150000):
    height, width, _ = image.shape
    scale = np.sqrt(pix_area_limit / (height * width))
    max_hight = int(scale * height)
    max_width = int(scale * width)
    if keep_original_proportion and scale > 1:
        return [height, width]
    else:
        return [max_hight, max_width]


def combine_images(img1, img2):
    height, width, _ = img1.shape
    if height > width:
        return np.concatenate((img1, np.zeros([height, 5, 3]), img2), axis=1)
    else:
        return np.concatenate((img1, np.zeros([5, width, 3]), img2), axis=0)


def flip_image(image, flip_ver=False):
    if flip_ver:
        return image[::-1, :, :]
    else:
        return image[:, ::-1, :]


def preserve_content_color(donor_image, recipient_image):
    YUV_color_space = cv2.COLOR_RGB2YUV
    RGB_color_space = cv2.COLOR_YUV2RGB
    donor_YUV = cv2.cvtColor(donor_image, YUV_color_space)
    recipient_YUV = cv2.cvtColor(recipient_image, YUV_color_space)
    lumina_chanel, _, _ = cv2.split(recipient_YUV)
    _, chromina1_chanel, chromina2_chanel = cv2.split(donor_YUV)
    rezalt = cv2.merge((lumina_chanel, chromina1_chanel, chromina2_chanel))
    rezalt = cv2.cvtColor(rezalt, RGB_color_space)  # .astype(np.uint8)
    rezalt = np.clip(rezalt, 0.01, 0.99)
    return rezalt


def prep(img):
    VGG_MEAN = [103.939, 116.779, 123.68]
    return (255 * img[..., ::-1] - VGG_MEAN).astype(np.float32)


def deprep(img):
    VGG_MEAN = [103.939, 116.779, 123.68]
    return np.clip(((img + VGG_MEAN)[..., ::-1]).astype(np.uint8), 0, 255)
