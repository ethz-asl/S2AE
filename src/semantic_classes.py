import numpy as np

class SemanticClasses(object):

    @staticmethod
    def map_nuscenes_label(label):
        noise = []
        vehicle = [1,2,3,4,5,6,7,8,9,10,11]
        living = [12,13,14,15,16,17,18,19]
        objects = [20,21,22,23]
        ground = [24,25,26,27]
        buildings = [28]
        vegetation = [29]
        other = [30,0]
        ego = [31]

        if label in noise:
            return 0
        elif label in vehicle:
            return 1
        elif label in living:
            return 2
        elif label in objects:
            return 3
        elif label in ground:
            return 4
        elif label in buildings:
            return 5
        elif label in vegetation:
            return 6
        elif label in other:
            return 7
        elif label in ego:
            return 8
        else:
            return 0
        
    @staticmethod
    def map_sem_kitti_label(label):
        unlabeled = [0, 1, 52, 99]
        car = [10, 252]
        bicycle = [11]
        other_vehicle = [13, 16, 20, 256, 257, 259]
        motorcycle = [15]
        truck = [18, 258]
        person = [30, 254]
        bicyclist = [31, 253]
        motorcyclist = [32, 255]
        road = [40, 60]
        parking = [44]
        sidewalk = [48]
        other_ground = [49]
        building = [50]
        fence = [14]
        vegetation = [70]
        trunk = [71]
        terrain = [72]
        pole = [80]
        traffic_sign = [81]
        
        if label in unlabeled:
            return 0
        elif label in car:
            return 1
        elif label in bicycle:
            return 2
        elif label in other_vehicle:
            return 5
        elif label in motorcycle:
            return 3
        elif label in truck:
            return 4
        elif label in person:
            return 6
        elif label in bicyclist:
            return 7
        elif label in motorcyclist:
            return 8
        elif label in road:
            return 9
        elif label in parking:
            return 10
        elif label in sidewalk:
            return 11
        elif label in other_ground:
            return 12
        elif label in building:
            return 13
        elif label in fence:
            return 14
        elif label in vegetation:
            return 15
        elif label in trunk:
            return 16
        elif label in terrain:
            return 17
        elif label in pole:
            return 18
        elif label in traffic_sign:
            return 19
        else:
            return 0

if __name__ == "__main__":
    pass