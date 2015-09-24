#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

class BatchIterator(object):
    """ Iterator for Reading Mini-Batches
    """

    def __init__(self,
                 input_file,
                 dictionary,
                 batch_size=128,
                 max_sequence_length=100):
        """
        :type input_file: file object
        :param input_file: input text file

        :type dictionary: Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs

        :type batch_size: int
        :param batch_size: number of sentences in one mini-batch (unless the end
                           of file is encountered earlier)

        :type max_sequence_length: int
        :param max_sequence_length: if not None, limit to sequences shorter than
                                    this
        """
        self.input_file = input_file
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.end_of_file = False

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next mini-batch read from the file.

        :rtype: tuple of numpy matrices
        :returns: two matrices - one contains the word IDs of each sequence
                  (0 after the last word), and the other contains a mask that
                  ís 1 after the last word
        """

        # If EOF was reach on previous call, but a mini-batch was returned,
        # rewind the file pointer now and raise StopIteration.
        if self.end_of_file:
            self.end_of_file = False
            self._reset()
            raise StopIteration

        sequences = []
        while True:
            line = self._readline()
            if line == '':
                break
            words = line.split()
            if not self.max_sequence_length is None and len(words) > self.max_sequence_length:
                continue
            sequences.append(self.dictionary.text_to_ids(words))
            if len(sequences) >= self.batch_size:
                return self._prepare_minibatch(sequences)

        # When end of file is reached, if no lines were read, rewind to first
        # line and raise StopIteration. If lines were read, return them and
        # raise StopIteration the next time this method is called.
        if len(sequences) == 0:
            self._reset()
            raise StopIteration
        else:
            self.end_of_file = True
            return self._prepare_minibatch(sequences)

    def _reset(self):
        self.input_file.seek(0)

    def _readline(self):
        return self.input_file.readline()

    def _prepare_minibatch(self, sequences):
        """Prepares a mini-batch for input to the neural network by transposing
        the sequences matrix and creating a mask matrix.

        The first dimensions of the returned matrix word_ids will be the time
        step, i.e. the index to a word in a sequence. In other words, the first
        row will contain the first word ID of each sequence, the second row the
        second word ID of each sequence, and so on. The rest of the matrix will
        be filled with zeros.

        The other returned matrix, mask, is the same size as word_ids, and will
        contain zeros where word_ids contains word IDs, and ones elsewhere
        (after sequence end).

        :type sequences: list of lists
        :param sequences: list of sequences, each of which is a list of word
                          IDs

        :rtype: tuple of numpy matrices
        :returns: two matrices - one contains the word IDs of each sequence
                  (0 after the last word), and the other contains a mask that
                  ís 1 after the last word
        """

        num_sequences = len(sequences)
        sequence_lengths = [len(s) for s in sequences]
        minibatch_length = numpy.max(sequence_lengths) + 1

        word_ids = numpy.zeros((minibatch_length, num_sequences)).astype('int64')
        mask = numpy.zeros((minibatch_length, num_sequences)).astype('float32')
        for i, sequence in enumerate(sequences):
            word_ids[:sequence_lengths[i], i] = sequence
            mask[:sequence_lengths[i] + 1, i] = 1.0

        return word_ids, mask

class OrderedBatchIterator(BatchIterator):
    """ Iterator for Reading Mini-Batches in Given Order
    """

    def __init__(self,
                 input_file,
                 dictionary,
                 line_starts,
                 batch_size=128,
                 max_sequence_length=100):
        """
        :type input_file: file object
        :param input_file: input text file

        :type dictionary: Dictionary
        :param dictionary: dictionary that provides mapping between words and
                           word IDs

        :type line_starts: numpy.ndarray
        :param line_starts: a list of start positions of the input sentences;
                            the sentences will be read in the order they appear
                            in this list

        :type batch_size: int
        :param batch_size: number of sentences in one mini-batch (unless the end
                           of file is encountered earlier)

        :type max_sequence_length: int
        :param max_sequence_length: if not None, limit to sequences shorter than
                                    this
        """

        super().__init__(input_file, dictionary, batch_size, max_sequence_length)

        self.line_starts = line_starts
        self.next_line = 0

    def _reset(self):
        self.next_line = 0

    def _readline(self):
        if self.next_line >= len(self.line_starts):
            return ''
        else:
            self.input_file.seek(self.line_starts[self.next_line])
            line = self.input_file.readline()
            self.next_line += 1
            return line