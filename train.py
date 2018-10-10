import time
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from utils import build_dict, build_dataset, batch_iter

# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")

    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")

    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")



parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
with open("args.pickle", "wb") as f:
    pickle.dump(args, f)

if not os.path.exists("saved_model"):
    os.mkdir("saved_model")
else:
    if args.with_model:
        old_model_checkpoint_path = open('saved_model/checkpoint', 'r')
        old_model_checkpoint_path = "".join(["saved_model/",old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])


print("Building dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", args.toy)
print("Loading training dataset...")
train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, args.toy)


with tf.Session() as sess:
    model = Model(reversed_dict, article_max_len, summary_max_len, args)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if 'old_model_checkpoint_path' in globals():
        print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
        saver.restore(sess, old_model_checkpoint_path )

    batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

    print("\nIteration starts.")
    print("Number of batches per epoch :", num_batches_per_epoch)
    for batch_x, batch_y in batches:       
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]        
        batch_decoder_input = [[word_dict['<s>']] + [x] for x in batch_y]
        batch_decoder_len = [len([y for y in x if y != 0]) for x in batch_decoder_input]      
        batch_decoder_output = [[x] + [word_dict['</s>']] for x in batch_y]

        batch_decoder_input = [d + (summary_max_len - len(d)) * [word_dict['<padding>']] for d in batch_decoder_input]
        batch_decoder_output = [d + (summary_max_len - len(d)) * [word_dict['<padding>']] for d in batch_decoder_output]        

        train_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
            model.decoder_input: batch_decoder_input,
            model.decoder_len: batch_decoder_len,
            model.decoder_target: batch_decoder_output
        }

        _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 1000 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % num_batches_per_epoch == 0:
            hours, rem = divmod(time.perf_counter() - start, 3600)
            minutes, seconds = divmod(rem, 60)
            saver.save(sess, "./saved_model/model.ckpt", global_step=step)
            print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
            "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")
