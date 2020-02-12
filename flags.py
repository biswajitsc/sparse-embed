import absl

# gb_limit = int(3e11 / 8) # 300 gb divided into 8 processes
# resource.setrlimit(resource.RLIMIT_AS, (gb_limit, gb_limit))

# Define all necessary hyperparameters as flags
flags = absl.flags

# The same flag is used for the various kinds of l1 parameters
# used in the different l1 weighting schemes
flags.DEFINE_float(name='l1_parameter',
                   help='The value of the l1_parameter',
                   default=0.0)
flags.DEFINE_float(name='zero_threshold',
                   help='Threshold below which values will be considered zero',
                   default=1e-8)
flags.DEFINE_float(name='learning_rate', help='Learning rate', default=1e-2)
flags.DEFINE_float(name='momentum',
                   help='Momentum value if using momentum optimizer',
                   default=0.7)
flags.DEFINE_float(name='l1_p_steps',
                   help='Number of steps to reach maximum weight',
                   default=5e4)

flags.DEFINE_integer(name='batch_size', help='Batch size', default=64)
# flags.DEFINE_integer(name='num_epochs', help='Number of training epochs', default=10000)
# Number of classes for Megaface should be 604854 + 1
flags.DEFINE_integer(name='num_classes', help='Number of training classes', default=1001)
flags.DEFINE_integer(name='embedding_size',
                     help='Embedding size',
                     default=1024)
flags.DEFINE_integer(name='test_size', help='Test size', default=2000)
flags.DEFINE_integer(name='decay_step', help='Number of steps after which to decay lr by 10', default=None)
flags.DEFINE_integer(name='max_steps', help='Number of steps to train', default=1000000)

flags.DEFINE_string(name='mobilenet_checkpoint_path',
                    help='Path to the checkpoint file for MobileNetV2',
                    default=None)
flags.DEFINE_string(name='model_dir',
                    help='Path to the model directory for checkpoints and summaries',
                    default='checkpoints/sparse_mobilenet/')
flags.DEFINE_string(name='data_dir',
                    help='Directory containing the training and validation data',
                    default='../imagenet/tfrecords')
flags.DEFINE_string(name='lfw_dir',
                    help='Directory containing the LFW data',
                    default='../msceleb1m')
flags.DEFINE_string(name='optimizer',
                    help="The optimizer to use. Options are 'sgd', 'mom' and 'adam'",
                    default='sgd')
flags.DEFINE_string(name='model',
                    help='Options are mobilenet, mobilefacenet, metric_learning',
                    default='mobilenet')
# Flag for the l1 weighting scheme to be used.
# Constant: constant l1 weight equal to l1_parameter
# Dynamic_1: l1_parameter / EMA(l1_norm)
# Dynamic_2: l1_weight += \lambda * (c - cl_loss)
flags.DEFINE_string(name='l1_weighing_scheme',
                    help='constant, dynamic_1, dynamic_2, dynamic_3',
                    default=None)
flags.DEFINE_string(name='sparsity_type',
                    help='Options are l1_norm, flops_sur',
                    default=None)
flags.DEFINE_string(name='final_activation',
                    help='The activation to use in the final layer producing the embeddings',
                    default=None)
flags.DEFINE_string(name='megaface_dir',
                    help='Root directory of the megaface tfrecords',
                    default='../megaface_distractors/tfrecords_official/')
flags.DEFINE_string(name='facescrub_dir',
                    help='Root directory of the facescrub tfrecords',
                    default='../facescrub/tfrecords_official/')
# flags.DEFINE_string(name='megaface_list',
#                     help='List of megaface images',
#                     default='../megaface_distractors/templatelists/megaface_features_list.json_10000_1')
# flags.DEFINE_string(name='facescrub_list',
#                     help='List of facescrub images',
#                     default='../megaface_distractors/templatelists/facescrub_features_list.json')


flags.DEFINE_boolean(name='restore_last_layer',
                     help='If True, the last layer will be restored when warm starting from '
                          'checkpoint. If checkpoint has a different number of output classes, '
                          'then set to False.',
                     default=True)
flags.DEFINE_boolean(name='evaluate',
                     help='If true then evaluates model while training',
                     default=False)

# Used by the evaluation scripts
flags.DEFINE_boolean(name='debug',
                     help='If True then inference will be stopped at 10000 steps',
                     default=False)
