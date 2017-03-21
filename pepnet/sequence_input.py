from .input_builder import InputBuilder

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
            conv_output_dim=16,
            conv_dropout=0.1,
            n_conv_layers=1,
            pool_size=3,
            pool_stride=2):
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
        builder = InputBuilder(
            name=self.name,
            length=self.length,
            n_symbols=self.n_symbols,
            encoding=self.encoding)

        if self.encoding == "index":
            assert self.embedding_dim > 0, \
                "Invalid embedding dim: %d" % self.embedding_dim
            builder.embedding(self.embedding_dim)

        if self.conv_filter_sizes:
            for i in range(self.n_conv_layers):
                builder.conv(
                    filter_sizes=self.conv_filter_sizes,
                    output_dim=self.conv_output_dim,
                    dropout=self.conv_dropout)
                # add max pooling for all layers before the last
                if i + 1 < self.n_conv_layers:
                    builder.local_max_pooling(size=self.pool_size, stride=self.pool_stride)
        return builder.value

