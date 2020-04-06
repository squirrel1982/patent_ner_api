# coding:utf-8
from base_model import BaseModel
from config import Config
import tensorflow as tf
from biLSTMCell import BiLSTMCell
from data_util import pad_sequences,ConfusionMatrix,get_chunks
from util import Progbar,minibatches_add_random_data,minibatches

class Patent_Ner_Model(BaseModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """
    def __init__(self, config):
        super(Patent_Ner_Model, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           config.vocab_tags.items()}
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder = tf.placeholder(
            tf.int32, shape = (None, Config.max_length, Config.n_features), name = 'input')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape = (None, Config.max_length), name = 'labels')
        self.mask_placeholder = tf.placeholder(
            tf.bool, shape = (None, Config.max_length), name = 'mask')
        self.dropout_placeholder = tf.placeholder(
            tf.float32, name = 'dropout')
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.mask_placeholder: mask_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embeddings = tf.nn.embedding_lookup(
            tf.Variable(self.config.embeddings),
            self.input_placeholder)
        embeddings = tf.reshape(
            embeddings, [-1, Config.max_length, Config.n_features* Config.embed_size])
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#pack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        cell = BiLSTMCell(Config.n_features * Config.embed_size, Config.hidden_size)
        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        with tf.variable_scope('Layer1'):
            U = tf.get_variable('U', (2*Config.hidden_size, Config.n_classes), initializer=tf.contrib.layers.xavier_initializer() )
            b2 = tf.get_variable('b2', (Config.n_classes), initializer=tf.constant_initializer(0) )

        #input_shape = tf.shape(x)
        #state = tf.zeros( (input_shape[0], Config.hidden_size) )
        ### END YOUR CODE

        with tf.variable_scope("RNN"):
            inputs_emb = tf.transpose(x, [1, 0, 2])
            # print '[2]:'+inputs_emb.shape
            inputs_emb = tf.reshape(inputs_emb, [-1, Config.embed_size*Config.n_features])
            # print '[3]:'+inputs_emb.shape
            inputs_emb = tf.split(inputs_emb, Config.max_length, 0)
            state = cell(inputs_emb)
            for state_tmp in state:
                state_dropout = tf.nn.dropout(state_tmp, dropout_rate)
                output_tmp = tf.matmul(state_dropout,U) + b2
                preds.append(output_tmp)
            ### END YOUR CODE

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines)
        preds = tf.stack(preds, axis=1)
        ### END YOUR CODE

        assert preds.get_shape().as_list() == [None, Config.max_length, Config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, Config.n_classes], preds.get_shape().as_list())
        self.logits = tf.reshape(preds,[-1,Config.max_length, Config.n_classes])
        return preds
    def output_4columns(self, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.config.helper.vectorize_4columns(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_,_ = self.predict_on_batch(*batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def add_loss_op(self,preds):
        """Defines the loss"""

        self.sequence_lengths = tf.reshape(self.labels_placeholder, [Config.batch_size, Config.max_length])
        self.sequence_lengths = tf.reduce_sum(tf.sign(self.sequence_lengths), reduction_indices=1)
        self.sequence_lengths = tf.cast(self.sequence_lengths, tf.int32)

        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            preds, self.labels_placeholder, self.sequence_lengths)
        self.trans_params = trans_params # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)
        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        return self.loss
    def add_training_op(self, loss):
        """Sets up the training Ops.b

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.config.helper.START, self.config.helper.END)
        return pad_sequences(examples, Config.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        print len(examples_raw),len(preds)
        assert len(examples_raw) == len(preds)


        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            #print len(labels_),len(labels)
            #如果长度不同，那么就往labels_后面补充O
            if len(labels_) != len(labels):
                diff = len(labels) - len(labels_)
                padding_tag = [Config.LBLS.index('O') for i in range(diff)]
                labels_.extend(padding_tag)
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret


    def predict_on_batch(self, inputs_batch, mask_batch):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        feed = self.create_feed_dict(inputs_batch, mask_batch=mask_batch,
                                     dropout=1.0)
        #还需要计算batch中每个句子的长度
        sequence_lengths =  1*mask_batch
        sequence_lengths = sequence_lengths.sum(axis=1)

        if 1==1:
            # get tag scores and transition params of CRF

            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=feed)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths


    def predict_on_batch_1(self, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = self.sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def fit(self, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            self.logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(train_examples, dev_set, dev_set_raw)
            if score > best_score:
                best_score = score
                if self.saver:
                    self.logger.info("New best score! Saving model in %s", Config.model_output)
                    self.saver.save(self.sess, Config.model_output)
            print("")
        return best_score

    def evaluate_4columns(self, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=Config.LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.output_4columns(examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(get_chunks(labels))
            pred = set(get_chunks(labels_))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)


    def run_epoch(self, train_examples, dev_set, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))

        for i, batch in enumerate(minibatches_add_random_data(train_examples, Config.batch_size)):
            loss = self.train_on_batch(*batch)
            prog.update(i + 1, [("train loss", loss)])
        print("")

        self.logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate_4columns(dev_set, dev_set_raw)
        self.logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        self.logger.debug("Token-level scores:\n" + token_cm.summary())
        self.logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1
    def run_epoch_1(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = Config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, Config.lr,
                    Config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.initialize_session()