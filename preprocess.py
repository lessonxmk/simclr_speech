import librosa
import glob
from absl import flags
from absl import app
import numpy as np
import os
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')
flags.DEFINE_string(
    'target_dir', None,
    'Directory where preprocessed dataset is stored.')
flags.DEFINE_string(
    'format', 'wav',
    'audio format.')
flags.DEFINE_integer(
    'rate', 16000,
    'audio rate.')
flags.DEFINE_float(
    'length', 2,
    'audio segment length.')


def main(argv):
    data_paths = glob.glob(FLAGS.data_dir + '/*.' + FLAGS.format)
    for p in tqdm(data_paths):
        wav_data, _ = librosa.load(p, FLAGS.rate)
        i = 0
        step = int(FLAGS.rate * FLAGS.length)
        while (i + step < len(wav_data)):
            seg = wav_data[i:i + step]
            mel = librosa.feature.melspectrogram(seg, FLAGS.rate)
            mel = np.log10(mel)
            np.save(FLAGS.target_dir + '/' + os.path.basename(p) + '-' + str(int(i / step)) + '.npy', mel)
            i += step


if __name__ == '__main__':
    app.run(main)
