from common import *
from discriminator import *
from generator import *


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint (
	generator_optimizer=generator_optimizer,
	discriminator_optimizer=discriminator_optimizer,
	generator=generator,
	discriminator=discriminator
)
                                 

log_dir="logs/"

summary_writer = tf.summary.create_file_writer (
	log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

@tf.function
def train_step(input_image, target, step):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(input_image, training=True)

		disc_real_output = discriminator([input_image, target], training=True)
		disc_generated_output = discriminator([input_image, gen_output], training=True)

		gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
		disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

	generator_gradients = gen_tape.gradient(gen_total_loss,
	generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss,
	discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_gradients,
	generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
	discriminator.trainable_variables))

	with summary_writer.as_default():
		tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
		tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
		tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
		tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
