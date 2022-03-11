import numpy as np

class SubSampler:

    def __init__(self, input_data, output_bw):
        self.input_data = input_data
        self.input_bw = self._compute_input_bw(input_data)
        self.output_bw = output_bw

    def _compute_input_bw(sefl, input_data):
        assert(input_data.ndim == 4)
        return input_data.shape[3] / 2

    def compute_output_data(self):
        return 42


if __name__ == '__main__':
    test_input = np.random.rand(10, 3, 400, 400)
    print('--- SubSampler Test Driver -------------------')
    print(f'Input matrix shape is: {test_input.shape}')

    sampler = SubSampler(test_input, 100)
    assert(sampler.input_bw == 200)

    print('Finished.')
