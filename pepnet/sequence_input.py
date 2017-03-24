from .helpers import conv, embedding, make_sequence_input, local_max_pooling

class SequenceInput(object):
    def __init__(
            self,
            name,
            length,
            n_symbols=21,
            encoding="index",
            embedding_dim=32,
            embedding_dropout=0,
            conv_filter_sizes=[],
            n_conv_layers=1,
            conv_output_dim=16,
            conv_dropout=0.1,
            pool_size=3,
            pool_stride=2):
        """
        Parameters
        ----------
        name : str
            Name of input sequence

        length : int
            Maximum length of sequence

        n_symbols : int
            Number of distinct symbols in sequences, default expects
            20 amino acids + 1 character for padding ('-')

        encoding : {"index", "onehot"}
            How are symbols represented: via integer indices or boolean
            vectors?

        embedding_dim : int
            How many dimensions in the symbol embedding
            (only used for index encoding)

        embedding_dropout : float
            What fraction of symbol representations are randomly set to 0
            during training?

        conv_filter_sizes : list of int
            Width of convolutional filters to apply to input vectors

        n_conv_layers : int
            Number of convolutional layers (with interleaving max pooling)
            to create

        conv_output_dim : int
            Number of filters per size of convolution

        conv_dropout : float
            Fraction of convolutional activations to randomly set to 0 during
            training

        pool_size : int
            If using more than one convolutional layer, how many timesteps
            from a previous layer get combined via maxpooling before the next
            layer.

        pool_stride : int
            If using more than one convolutional layer, stride of the max
            pooling.
        """
        self.name = name
        self.length = length
        self.encoding = encoding
        self.embedding_dim = embedding_dim
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_dropout = conv_dropout
        self.conv_output_dim = conv_output_dim
        self.n_conv_layers = n_conv_layers
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def build(self):
        input_object = make_sequence_input(
            encoding=self.encoding,
            name=self.name,
            length=self.length,
            n_symbols=self.n_symbols)

        if self.encoding == "index":
            assert self.embedding_dim > 0, \
                "Invalid embedding dim: %d" % self.embedding_dim
            value = embedding(input_object, embedding_dim=self.embedding_dim)
        else:
            value = input_object

        if self.conv_filter_sizes:
            for i in range(self.n_conv_layers):
                value = conv(
                    filter_sizes=self.conv_filter_sizes,
                    output_dim=self.conv_output_dim,
                    dropout=self.conv_dropout)
                # add max pooling for all layers before the last
                if i + 1 < self.n_conv_layers:
                    value = local_max_pooling(
                        size=self.pool_size,
                        stride=self.pool_stride)
        return input_object, value
