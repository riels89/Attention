import tensorflow as tf
import numpy as np

imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                          value=word_index["<PAD>"],
                                                          padding='post',
                                                          maxlen=256)
class transformer:

    def __init__(self):
        self.embedding_size = 512
        self.attention_size = 64

        self.X = tf.placeholder(tf.float32, shape=[None, 256], name="input")
        self.y = tf.placeholder(tf.int32, shape=(None, 1), name="labels")

    def transformer(self, X):
        # X = (n, t, 512)
        # (n, t, 64)
        Q = tf.matmul(X, tf.random.uniform([self.embedding_size, self.attention_size]))
        K = tf.matmul(X, tf.random.uniform([self.embedding_size, self.attention_size]))
        V = tf.matmul(X, tf.random.uniform([self.embedding_size, self.attention_size]))

        # (n, t, 64) * (n, t, 64).T = (n, t, n, t)
        score = tf.matmul(Q, tf.transpose(K))


        softmax = tf.nn.softmax(tf.divide(score, np.sqrt(self.attention_size)), axis=-1)

        value_vec = tf.matmul(softmax, V)

        return value_vec

    def createModel(self):

        embedding = tf.layers.dense(self.X, self.embedding_size)
        t1 = self.transformer(embedding)

        fc = tf.layers.dense(t1, 1)

        return fc

model = transformer()
fc = model.createModel()

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(tf.global_variables_initializer())

  output = sess.run([fc], {model.X: train_data, model.y: train_labels[:, np.newaxis]})
  print(output)
