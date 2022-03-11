import numpy as np

class SubSampler:

    def __init__(self, input_data, output_bw):
        self.input_data = input_data
        self.input_bw = self._compute_input_bw(input_data)
        self.output_bw = output_bw

    def _compute_input_bw(sefl, input_data):
        # assert input_data.shape
        return 42

    def compute_output_data(self):
        return 42


if __name__ == '__main__':
    print('foo')
