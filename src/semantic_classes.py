import numpy as np

class SemanticClasses(object):

    def map_nuscenes_label_old(label):
        noise = [0]
        vehicle = [1,2,3,4,5,6,7,8,9,10,11]
        living = [12,13,14,15,16,17,18,19]
        objects = [20,21,22,23]
        ground = [24,25,26,27]
        buildings = [28]
        vegetation = [29]
        other = [30]
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

    def map_nuscenes_label(label):
        noise = [0, 1, 5, 10, 11, 13, 29, 31]
        living = [2,3,4,6,7,8]
        vehicle = [14,15,16,17,18,19,20,21,22,23]    
        road = [24]
        ground = [25,26,27]
        manmade = [9,12,28]
        vegetation = [30]

        if label in noise:
            return 0
        elif label in vehicle:
            return 1
        elif label in living:
            return 2
        elif label in road:
            return 3
        elif label in ground:
            return 4
        elif label in manmade:
            return 5
        elif label in vegetation:
            return 6    
        else:
            return 0

    def map_waymo_to_nuscenes_label(label):
        noise = [0]
        living = [7]
        vehicle = [1,2,3,4,5,6,12,13]
        road = [18,19]
        ground = [17,20,21,22]
        manmade = [8,9,10,11,14]
        vegetation = [15,16]
        label = int(label)

        if label in noise:
            return 0
        elif label in vehicle:
            return 1
        elif label in living:
            return 2
        elif label in road:
            return 3
        elif label in ground:
            return 4
        elif label in manmade:
            return 5
        elif label in vegetation:
            return 6    
        else:
            return 0

    def map_poss_to_nuscenes_label(label):
        noise = [0,22]
        living = [4,5]
        vehicle = [6,7,21]
        road = []
        ground = []
        manmade = [10,11,12,13,14,15,16,17]
        vegetation = [8,9]

        if label in noise:
            return 0
        elif label in vehicle:
            return 1
        elif label in living:
            return 2
        elif label in road:
            return 3
        elif label in ground:
            return 4
        elif label in manmade:
            return 5
        elif label in vegetation:
            return 6    
        else:
            return 0

    def map_kitti_to_nuscenes_label(label):
        noise = [0,1,44,99]
        living = [30,254]
        vehicle = [10,11,13,15,16,18,20,31,32,252,253,255,256,257,258,259]
        road = [40,60]
        ground = [40,48,49,72]
        manmade = [50,51,52,80,81]
        vegetation = [70, 71]
        label = int(label)

        if label in noise:
            return 0
        elif label in vehicle:
            return 1
        elif label in living:
            return 2
        elif label in road:
            return 3
        elif label in ground:
            return 4
        elif label in manmade:
            return 5
        elif label in vegetation:
            return 6    
        else:
            if label >= 0:
                print(f'WARNING CLASS LABEL {label} not handled!!!!')
            return 0    

    def map_nuscenes_label_16(label):                       
        noise = [1, 5, 6, 8, 10, 11, 13, 19, 20, 0, 29, 31]
        barrier = [9]
        bicycle = [14]
        bus = [15, 16]
        car = [17]
        construction_vehicle = [18]
        motorcycle = [21]
        pedestrian = [2, 3, 4, 6]
        traffic_cone = [12]
        trailer = [22]
        truck = [23]
        driveable_surface = [24]
        other_flat = [25]
        sidewalk = [26]
        terrain = [27]
        manmade = [28]
        vegetation = [30]

        if label in noise:
            return 0
        elif label in barrier:
            return 1
        elif label in bicycle:
            return 2
        elif label in bus:
            return 3
        elif label in car:
            return 4
        elif label in construction_vehicle:
            return 5
        elif label in motorcycle:
            return 6
        elif label in pedestrian:
            return 7
        elif label in traffic_cone:
            return 8
        elif label in trailer:
            return 9
        elif label in truck:
            return 10
        elif label in driveable_surface:
            return 11
        elif label in other_flat:
            return 12
        elif label in sidewalk:
            return 13
        elif label in terrain:
            return 14
        elif label in manmade:
            return 15
        elif label in vegetation:
            return 16    
        else:
            return 0

if __name__ == "__main__":
    pass