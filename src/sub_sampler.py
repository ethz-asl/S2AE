import numpy as np

from sphere import Sphere
from dh_grid import DHGrid
from tqdm.auto import tqdm

class SubSampler:

    def __init__(self, input_data, output_bw):
        self.input_data = input_data
        self.input_bw = self._compute_input_bw(input_data)
        self.output_bw = output_bw

    def _compute_input_bw(sefl, input_data):
        assert(input_data.ndim == 4)
        return int(input_data.shape[3] / 2)

    def compute_output_data(self):
        n_clouds = self.input_data.shape[0]
        n_clouds = 10
        out = np.zeros((n_clouds, 3, 2*self.output_bw, 2*self.output_bw))
        for i in tqdm(range(n_clouds)):
            grid, _ = DHGrid.CreateGrid(self.output_bw)
            sph = Sphere(bw = self.input_bw, features = self.input_data[i, :, :, :])
            out[i, :, :, :] = sph.sampleUsingGrid(grid)
        return out


if __name__ == '__main__':
    test_input = np.random.rand(10, 3, 400, 400)
    print('--- SubSampler Test Driver -------------------')
    print(f'Input matrix shape is: {test_input.shape}')

    sampler = SubSampler(test_input, 100)
    assert(sampler.input_bw == 200)

    sampled = sampler.compute_output_data()
    print(f'sampled shape is {sampled.shape}')


    print('Finished.')
