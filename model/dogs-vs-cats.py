# building powerful image classification models using very little data
# image dataset path => C:\dogs-vs-cats

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=0.4,     # rotation degree
    width_shift_range=0.2,  # lateral move horizontally
    height_shift_range=0.2, # lateral move vertically
    rescale=.1 / 255,       # RGB rate rescaling 0 ~ 255 => 0 ~ 1
    shear_range=0.2,        # shearing transformation range
    zoom_range=0.2,         # zoom and out
    horizontal_flip=True,   # if True => flip image by 50%
    fill_mode="nearest"     # method of filling space when "rotated" "moved" "zoomed"
)

# other parameters => https://keras.io/preprocessing/image

# PIL
img = load_img("C:/dogs-vs-cats/train/cats/cat.0.jpg")
# size => 3(RGB) x 150(width) x 150(height)
x = img_to_array(img)
# size => 1 x 3(RGB) x 150(width) x 150(height)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir="C:/dogs-vs-cats/train/cats-preview",
                          save_prefix="cat",
                          save_format="jpeg"):
    i += 1
    if i > 20:
        break

