from gan import *
from common import *
import tensorflow as tf
from PIL import Image

# Load last checkpoint 
checkpoint.restore(
    tf.train.latest_checkpoint(checkpoint_dir)
)

def detectionLoad(img_path):
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image)
    
    image /= 255 # Normalizing image
    
    image = tf.cast(image, tf.float32)
    
    return image
    
def genimage(model, test_input):

    genimg = model(test_input, training=True)
    
    img = plt.imshow(genimg[0])
    plt.axis('off')
    
    files = os.listdir(r"./info/trainingImages/")
    plt.savefig(fr"./info/trainingImages/{len(files)}.png" , bbox_inches='tight')


def detectionTest():
    PATH = r"detect/"
    detect_dataset = tf.data.Dataset.list_files(PATH+"*.jpg")
    detect_dataset = detect_dataset.map(detectionLoad, num_parallel_calls=tf.data.AUTOTUNE)
    detect_dataset = detect_dataset.shuffle(400)
    detect_dataset = detect_dataset.batch(1)
    
    for inp in detect_dataset:
        genimage(generator, inp)
    
def images_resize():
    PATH = r"detect/"
    for file in os.listdir(PATH):
        img = Image.open(PATH+file)
        shape = img.size
        img = img.resize((256, 256))
        img.save(PATH+file)

if __name__ == "__main__":
    images_resize()
    detectionTest()
    
