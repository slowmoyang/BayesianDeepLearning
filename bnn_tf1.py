import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from dataset import get_qgjets_paths
from dataset import load_dataset
from dataset import build_input_pipeline


def build_model(features=None):
    values = [] if features is None else [features]
    with tf.compat.v1.name_scope("bayesian_neural_network", values=values):
        bnn = tf.keras.Sequential([
            tfp.layers.DenseFlipout(128, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(128, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(128, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(2)
        ])

    if features is None:
        return bnn
    else:
        logits = bnn(features)
        labels_dist = tfd.Categorical(logits=logits)
        predictions = tf.argmax(input=logits, axis=1)
        return bnn, logits, labels_dist, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-pt', default=100, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('-T', '--num-samples', default=1000, type=int)
    parser.add_argument('-e', '--num-epochs', default=100, type=int)
    parser.add_argument('-d', '--out-dir', default='/tmp/')
    parser.add_argument('-n', '--out-name')
    args = parser.parse_args()

    max_pt = int(1.1 * args.min_pt)
    if args.out_name is None:
        args.out_name = 'bnn_pt-{}-{}'.format(args.min_pt, max_pt)

    paths = get_qgjets_paths(args.min_pt)
    train_set = load_dataset(paths['training'])
    valid_set = load_dataset(paths['validation'])
    test_set = load_dataset(paths['test'])

    elements, handle, iterators = build_input_pipeline(
        train_set, valid_set, test_set, batch_size=args.batch_size)
    # just unpack
    features, labels = elements
    train_iter, valid_iter, test_iter = iterators

    # Build model
    bnn, logits, labels_dist, predictions = build_model(features)

    # Loss function
    # negative log likelihood
    nll = -tf.reduce_mean(input_tensor=labels_dist.log_prob(labels))
    # Kullback-Leibler dievergence
    kld = sum(bnn.losses) / len(train_set[0])
    # negative ELBO (evidence lower bound)
    loss = nll + kld

    # metric
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions)

    with tf.compat.v1.name_scope("train"):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=args.learning_rate)
        train_op = optimizer.minimize(loss)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        valid_handle = sess.run(valid_iter.string_handle())
        test_handle = sess.run(test_iter.string_handle())

        # Validation
        loss_value, accuracy_value = sess.run(
            fetches=[loss, accuracy],
            feed_dict={handle: valid_handle})

        print("\n[Validation] Epoch: {:>3d} / 100 @ Loss: {:.3f} @ Accuracy: {:.3f}\n".format(
            0, loss_value, accuracy_value))

        for epoch in range(1, args.num_epochs + 1):
            # Training
            sess.run(train_iter.initializer)
            train_handle = sess.run(train_iter.string_handle())
            step = 0
            while True:
                step += 1
                try:
                    _ = sess.run(fetches=[train_op, accuracy_update_op],
                                 feed_dict={handle: train_handle})

                    if step % 100 == 0:
                        loss_value, accuracy_value = sess.run(
                            fetches=[loss, accuracy],
                            feed_dict={handle: train_handle})

                        print("[Training] Epoch: {:>3d} / 100 @ Loss: {:.3f} @ Accuracy: {:.3f}".format(
                            epoch, loss_value, accuracy_value))
                except tf.compat.v1.errors.OutOfRangeError:
                    break

            # Validation
            loss_value, accuracy_value = sess.run(
                fetches=[loss, accuracy],
                feed_dict={handle: valid_handle})

            print("\n[Validation] Epoch: {:>3d} / 100 Loss: {:.3f} Accuracy: {:.3f}\n".format(
                epoch, loss_value, accuracy_value))

        print("@" * 10 + " END " + "@" * 10)

        y_true = sess.run(fetches=labels, feed_dict={handle: test_handle})

        def sample():
            return sess.run(fetches=[labels_dist.sample(), labels_dist.probs],
                            feed_dict={handle: test_handle})
        result = [sample() for _ in range(args.num_samples)]
        pred_samples, prob_samples = zip(*result)
        
        pred_samples = np.stack(prob_samples)
        prob_samples = np.stack(prob_samples)

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez(out_path,
             y_true=y_true,
             prob_samples=prob_samples,
             pred_samples=pred_samples)


if __name__ == '__main__':
    main()
