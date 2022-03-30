import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from RIM_model import RIM

import tensorflow as tf




flags = tf.app.flags
flags.DEFINE_integer("epoch", 0, "Epoch to train [0], default ==100")
flags.DEFINE_integer("iteration", 100000, "Iteration to train for each level of the hierarhcy. Either epoch or iteration need to be zero [0]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

# dataset name
flags.DEFINE_string("dataset", "02691156_train_vox", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "RIM_checkpoint/02691156/check", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data/02691156_airplane/", "Root directory of dataset [data]")
flags.DEFINE_string("result_dir", "RIM_checkpoint/02691156/result", "Directory name to save the image samples [samples]")

flags.DEFINE_integer("branch_num", 8, "num of parts of the last level of the hierarchy")
flags.DEFINE_integer("points_per_shape", 8192, "num of points per shape [32768]")
flags.DEFINE_integer("real_size", 64, "output point-value voxel grid size in training [64]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("recon", True, "(in testing mode) True for outputing reconstructed shape with colored segmentation [False]")
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.result_dir):
		os.makedirs(FLAGS.result_dir)

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		imseg = RIM(
				sess,
				FLAGS.real_size,
				FLAGS.points_per_shape,
				is_training = FLAGS.train,
				dataset_name=FLAGS.dataset,
				checkpoint_dir=FLAGS.checkpoint_dir,
				result_dir=FLAGS.result_dir,
				data_dir=FLAGS.data_dir,
				branch_num=FLAGS.branch_num)

		if FLAGS.train:
			imseg.train(FLAGS)
		else:
			if FLAGS.recon:
				imseg.test_dae(FLAGS) #output reconstructed shape with colored segmentation

if __name__ == '__main__':
	tf.app.run()
