import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
from absl import flags
from absl import app
import logging

import dataloader
import augment
import model
from larsOptimizer import LARS

FLAGS = flags.FLAGS
flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float(
    'weight_decay', 1e-6,
    'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_steps', 1000,
    'Number of steps between checkpoints/summaries.')

flags.DEFINE_integer(
    'logging_steps', 50,
    'Number of steps between loggings.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for continued training or fine-tuning.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'log_dir', None,
    'Directory where log is stored.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'head_proj_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'head_proj_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_nlh_layers', 1,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'load_checkpoint_step', None,
    'checkpoint step to load.')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def loadDataPath(format='wav', *path):
    dataPath = []
    for p in path:
        dataPath.append(glob.glob(p) + '.' + format)
    return dataPath()


def Augment(x, *operations):
    availableOperation = {
        'crop': augment.crop,
        'gwn': augment.Gaussian_white_noise,
    }
    for op in operations:
        x = availableOperation[op](x)
    return x


def initial_optim(param, optimizer='lars', lr=0.3, momentum=0.9, weight_decay=1e-6):
    if (optimizer == 'lars'):
        return LARS(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif (optimizer == 'adam'):
        return optim.Adam(param, lr=lr, weight_decay=weight_decay)
    elif (optimizer == 'momentum'):
        return optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)


def run(argv):
    logger = get_logger(FLAGS.log_dir)
    train_data = loadDataPath()
    train_data = dataloader.DataSet(train_data)
    train_loader = DataLoader(train_data, batch_size=FLAGS.train_batch_size, shuffle=True)
    checkpoints = glob.glob(FLAGS.checkpoint + '*.pth')
    if (len(checkpoints) > 0 and FLAGS.load_checkpoint_step is not None):
        encoder = model.Encoder()
        projectionHead = model.projectionHead()
        encoder.load_state_dict(
            torch.load(FLAGS.checkpoint + '-encoder-' + str(FLAGS.load_checkpoint_step) + '.pth'))
        projectionHead.load_state_dict(
            torch.load(FLAGS.checkpoint + '-projectionHead-' + str(FLAGS.load_checkpoint_step) + '.pth'))
    else:
        encoder = model.Encoder()
        projectionHead = None
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    criterion = model.contrastiveLoss(temperature=FLAGS.temperature, hidden_norm=FLAGS.hidden_norm)
    optimizerE = initial_optim(encoder.parameters(), optimizer=FLAGS.optimizer, lr=FLAGS.learning_rate,
                               momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    optimizerP = initial_optim(projectionHead.parameters(), optimizer=FLAGS.optimizer, lr=FLAGS.learning_rate,
                               momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    logger.info('Start training')
    step = 0
    for epoch in range(FLAGS.train_epochs):
        for _, x in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()

            x1 = Augment(x)
            x2 = Augment(x)
            x = torch.cat((x1, x2), 0)

            representation = encoder(x)
            if (projectionHead is None):
                projectionHead = model.projectionHead(x.shape, FLAGS.head_proj_dim, FLAGS.head_proj_mode)
                if torch.cuda.is_available():
                    projectionHead = projectionHead.cuda()
            out = projectionHead(representation)

            loss = criterion(out)
            optimizerE.zero_grad()
            optimizerP.zero_grad()
            loss.backward()
            optimizerE.step()
            optimizerP.step()

            step += 1
            if (step % FLAGS.logging_steps):
                logger.info('step:{},loss:{}'.format(step, loss.data.item()))
            if (step % FLAGS.checkpoint_steps == 0):
                logger.info('saving checkpoint-{}'.format(step))
                torch.save(encoder.state_dict(), FLAGS.checkpoint + '-encoder-' + str(step) + '.pth')
                torch.save(projectionHead.state_dict(), FLAGS.checkpoint + '-projectionHead-' + str(step) + '.pth')


if __name__ == '__main__':
    app.run(run())
