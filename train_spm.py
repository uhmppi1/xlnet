from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sentencepiece as spm
import tensorflow as tf
from absl import app
from absl import flags
# flags = tf.app.flags
import absl.logging as _logging  # pylint: disable=unused-import


flags.DEFINE_string("spm_model_prefix", default="sp10m.cased.v3",
                    help="spm model_prefix")
flags.DEFINE_integer("spm_vocab_size", default=500,
                     help="vocab_size for spm")
flags.DEFINE_string("input_spm", default="corpus/spm_data/aaa.txt",
                    help="Input file glob.")

FLAGS = flags.FLAGS

def main(unused_argv):
    print(FLAGS.input_spm)
    spm.SentencePieceTrainer.Train('--input=%s '
                                   '--model_prefix=%s '
                                   '--vocab_size=%d '
                                   '--character_coverage=1.0 '
                                   '--model_type=unigram '
                                   '--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> '
                                   '--user_defined_symbols=<eop>,.,(,),",-,–,£,€ '
                                   '--shuffle_input_sentence '
                                   '--input_sentence_size=10000000 '
                                   % (FLAGS.input_spm, FLAGS.spm_model_prefix, FLAGS.spm_vocab_size))

if __name__ == "__main__":
    app.run(main)
