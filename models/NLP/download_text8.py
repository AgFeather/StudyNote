import os
import numpy as np
import random
import string
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve



url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
	"""download a file if not present, and make sure it's the reight size"""
	if not os.path.exists(filename):
		filename, _ = urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('found and verified %s' % filename)
	else:
		print(statinfo.st_size)
		print('failed to verify')
	return filename

def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		name = f.namelist()[0]
		data = tf.compat.as_str(f.read(name))
	return data

#untility functions to map characters to vocabulary IDs and back
vocabulary_size = len(string.ascii_lowercase) + 1 #[a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])
def char2id(char):
	if char in string.ascii_lowercase:
		return ord(char) - first_letter + 1
	elif char == ' ':
		return 0
	else:
		print('unexpected character: %s' % char)
		return 0

def id2char(id):
	if id > 0:
		return chr(id + first_letter - 1)
	else:
		return ' '

def characters(probabilities):
	"""trun a one-hot encoding or a probability distribution over the possible
	characters batch into its (most likey) character representation"""
	return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
	"""convert a sequence of batches back into their (most likely) string
	representation."""
	s = [''] * batches[0].shape[0]
	for b in batches:
		s = [''.join(x) for x in zip(s, characters(b))]
	return s


def logprob(predictions, labels):
	"""Log-probability of the true labels in a predicted batch."""
	predictions[predictions < 1e-10] = 1e-10
	return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
	"""sample one element from a distribution assumed to be
	an array of normalized probabilities"""
	r = random.uniform(0, 1)
	s = 0
	for i in range(len(distribution)):
		s += distribution[i]
		if s >= r:
			return i
	return len(distribution)

def sample(prediction):
	#turn a (column) prediction into one-hot encoded samples
	p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
	p[0, sample_distribution(prediction[0])] = 1.0

def random_distribution():
	#generate a random column of probabilities.
	b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
	return b / np.sum(b, 1)[:,None]







if __name__ == '__main__':
	filename = maybe_download('text8.zip', 31344016)
	text = read_data(filename)# type(text) = string
	print('Data size %d' % len(text))

	#create a small validation set
	valid_size = 1000
	valid_text = text[:valid_size]
	train_text = text[valid_size:]
	train_size = len(train_text)
	print(train_size, train_text[:64])
	print(valid_size, valid_text[:64])

	#test char2id, id2char
	print(char2id('a'), char2id('z'), char2id(' '))
	print(id2char(1), id2char(26), id2char(0))
