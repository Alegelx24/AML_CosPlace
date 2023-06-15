
#The following code contains the processing useful to create synthetically 
# a night style dataset starting from a day one.

import numpy as np
from PIL import Image, ImageEnhance
import imageio
import os
import random
import cv2

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def FDA_transform(source_img, target_img, weight):

    im_src  = Image.open(source_img).convert('RGB')
    im_trg = Image.open(target_img).convert('RGB')

    im_src = im_src.resize( (640,480), Image.BICUBIC )
    im_trg = im_trg.resize( (640,480), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, weight )

    return src_in_trg.transpose((1,2,0))

def FDA_database_transform2(database_path, queries_path, output_dir, weight= 0.01):

    queries_list= os.listdir(queries_path)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root, dirs, files in os.walk(database_path):
        for filename in files:
           # if random.randint(1, 59) == 1:  # Apply FDA to every 59th image as sampling 
                query_index = int(random.uniform(0, 104))
                src_in_trg = FDA_transform(os.path.join(root, filename), os.path.join(queries_path, queries_list[query_index]), weight)
                output_filename = os.path.join(output_dir, filename)
                imageio.imwrite(output_filename, src_in_trg)

#add fake light inside images
def add_fake_light(image):

    width, height = image.size
    image_array = np.array(image)

    # Generate a mask with the same size as the image
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    num_lights = 10

    for _ in range(num_lights):
        # Generate random coordinates for the light
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        # Set the color of the light
        light_color = (255, 255, 255)  # Adjust the RGB values as per your needs
        # Set the radius and intensity of the light
        radius = np.random.randint(5, 20)
        intensity = np.random.randint(150, 300)
        # Create a circular gradient for the light
        cv2.circle(mask, (x, y), radius, light_color, -1, cv2.LINE_AA)
        # Add the light to the image array with the specified intensity
        image_array += intensity * mask

    image_array = np.clip(image_array, 0, 255)
    result_image = Image.fromarray(image_array)

    return result_image

def convert_all_to_night_V2(folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for root, _, files in os.walk(folder_path):
        for filename in files:
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)

            # Reduce brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.6)  # Adjust the enhancement factor as per your needs
            # Add blue overlay
            width, height = image.size
            blue_overlay = Image.new('RGB', (width, height), (0, 0, 50))  # Adjust the RGB values as per your needs
            image = Image.blend(image, blue_overlay, 0.35)  # Adjust the blend factor as per your needs
            # Desaturate the image
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1)  # Adjust the enhancement factor as per your needs
            # Add noise
            noise = np.random.normal(loc=0, scale=10, size=(height, width, 3)).astype(np.uint8)  # Adjust the scale as per your needs
            noise_image = Image.fromarray(noise, 'RGB')
            image = Image.blend(image, noise_image, 0.05)  # Adjust the blend factor as per your needs
            # Add fake light effect
            image = add_fake_light(image)

            output_filename = os.path.join(output_folder_path, filename)
            image.save(output_filename)

    print("Conversion to night completed successfully!")





if __name__ == "__main__":

    FDA_database_transform2("folderpath/of/source/images","folderpath/of/style/target/images","/output/folderpath", 0.01)
    convert_all_to_night_V2("/output/folderpath/previously_inserted","converted/image/folder/output/path")


