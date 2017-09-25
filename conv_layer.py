class ConvLayer:
    def __init__(self, filter_size, channel_size, pool_size):
        self.filter_size = filter_size
        self.channel_size = channel_size
        self.pool_size = pool_size

    def __str__(self):
        return '({}){}c-{}p'.format(self.channel_size, self.filter_size, self.pool_size)
