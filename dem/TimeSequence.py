import csv
import os
import taichi as ti
import numpy as np
from TypeDefine import *
@ti.dataclass
class ForceWithID:
    id: Integer
    force: Vector3

@ti.kernel
def apply_force_with_id(grain_force:ti.template(), grain_id_to_index:ti.template(), 
                        force_with_id:ti.template(), count:ti.template()):
    for i in range(count[None]):
        shiftId = force_with_id[i].id - 1
        particle_index = grain_id_to_index[shiftId]
        grain_force[particle_index] += force_with_id[i].force
        # print(f'force = {force_with_id[i].force}, id = {force_with_id[i].id}, index = {grain_id_to_index[force_with_id[i].id - 1]}')


class TimeSequence:
    def __init__(self, folder:str):
        self.folder = folder
        self.ranges = []
        self.RangeIndex = -1
        self.rangeExist = False
        self.grain_force_with_ID = None
        self.grain_force_size = ti.field(Integer, shape = ())
        # find all filenames in folder
        for root, directories, file in os.walk(folder):
            for f in file:
                if(f.endswith(".csv") and f.startswith("SGF")):
                    f = os.path.basename(f)
                    short = f.removeprefix("SGF").removesuffix(".csv")
                    frames = short.split('-')
                    begin, end = int(frames[0]), int(frames[1])
                    self.ranges.append((begin, end))
        
    def add_force(self, frame, dem_grain_force, dem_grain_id_to_index):
        '''
            add time sequence force to grain.force
        '''
        if(self.RangeIndex >= len(self.ranges)):
            print("No more force range")
            return
        apply = False
        if(not self.rangeExist):
            next = self.RangeIndex + 1
            if(next >= len(self.ranges)):
                print("No more force range")
                return
            nextRange = self.ranges[next]
            if frame >= nextRange[0] and frame <= nextRange[1] :
                self.RangeIndex = next
                self.rangeExist = True
                apply = True
                # we need to read the new file
                self.read_new_SGF()
            else:
                self.rangeExist = False
                apply = False
        else:
            currentRamge = self.ranges[self.RangeIndex]
            if frame >= currentRamge[0] and frame <= currentRamge[1]:
                apply = True
            else: # go to next range
                next = self.RangeIndex + 1
                if(next >= len(self.ranges)):
                    print("No more force range")
                    return
                nextRange = self.ranges[next]
                if frame >= nextRange[0] and frame <= nextRange[1] :
                    self.RangeIndex = next
                    self.rangeExist = True
                    apply = True
                    # we need to read the new file
                    self.read_new_SGF()
                else:
                    self.rangeExist = False
                    apply = False
                
        if(apply):
            apply_force_with_id(dem_grain_force, dem_grain_id_to_index, self.grain_force_with_ID, self.grain_force_size)


    def read_new_SGF(self):
        basename = "/SGF" + str(self.ranges[self.RangeIndex][0]) + "-" + str(self.ranges[self.RangeIndex][1]) + ".csv"
        filename = self.folder + basename
        print('switch to new SGF file: ' + basename)
        np_grain_force_ID = []
        np_grain_force = []
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            first = True
            for row in reader:
                if(first):
                    first = False
                    continue
                np_grain_force_ID.append(int(row[0]))
                np_grain_force.append((float(row[1]), float(row[2]), float(row[3])))
        if(self.grain_force_with_ID == None or self.grain_force_with_ID.shape[0] < len(np_grain_force)):
            self.grain_force_with_ID = ForceWithID.field(shape = len(np_grain_force))
            print(f"grain_force_with_ID size = {len(np_grain_force)}")
        self.grain_force_with_ID.id.from_numpy(np.array(np_grain_force_ID))
        self.grain_force_with_ID.force.from_numpy(np.array(np_grain_force))
        self.grain_force_size[None] = len(np_grain_force)
        
if __name__ == "__main__":
    ti.init(arch = ti.gpu)
    t = TimeSequence("input_data/SGF")
    t.rangeExist = True
    t.rangeIndex = 0
    t.read_new_SGF()