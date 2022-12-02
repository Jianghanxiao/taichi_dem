from math import pi
import taichi as ti
import taichi.math as tm
import os
import numpy as np
from TypeDefine import *
from Utils import *
from DemConfig import *
from DEMSolverStatistics import *
from PrefixSumExecutor import *
@ti.dataclass
class Range:
    offset:Integer
    count:Integer
    current:Integer

# @ti.data_oriented
# class PrefixSumExecutor:
#     def __init__(self):
#         self.tree:ti.SNode = None
#         self.temp:ti.StructField = None

#     def _resize_temp(self, n):
#         ti.sync()
#         if(self.tree != None):
#             if(self.temp.shape[0] >= n): return
#             else: pass
#                 # self.tree.destroy()
#         # ti.sync()
#         # realloc
#         print(f"resize_prefix_sum_temp:{n}")
#         fb = ti.FieldsBuilder()
#         self.temp = ti.field(Integer)
#         fb.dense(ti.i, n).place(self.temp)
#         self.tree = fb.finalize()
    
#     @ti.kernel
#     def serial(self, output:ti.template(), input:ti.template()):
#         n = input.shape[0]
#         output[0] = 0
#         ti.loop_config(serialize=True)
#         for i in range(1, n): 
#             output[i] = output[i - 1] + input[i - 1]

#     @ti.kernel
#     def _down(self, d:Integer, 
#                     n:Integer,
#                     offset:ti.template(),
#                     output:ti.template()):
#             for i in range(n):
#                 if(i < d):
#                     ai = offset*(2*i+1)-1
#                     bi = offset*(2*i+2)-1
#                     output[bi] += output[ai]
    
#     @ti.kernel
#     def _up(self,
#             d:Integer, 
#             n:Integer,
#             offset:ti.template(),
#             output:ti.template()):
#         for i in range(n):
#             if(i < d):
#                 ai = offset*(2*i+1)-1
#                 bi = offset*(2*i+2)-1
#                 tmp = output[ai]
#                 output[ai] = output[bi]
#                 output[bi] += tmp
    
#     @ti.kernel
#     def _copy(self, n:Integer,
#               output:ti.template(),
#               input:ti.template()):
#         for i in range(n): output[i] = input[i]
#     @ti.kernel
#     def _copy_and_clear(self, n:Integer, npad:Integer, temp:ti.template(), input:ti.template()):
#         for i in range(n): temp[i] = input[i]
#         for i in range(n, npad): temp[i] = 0

#     # def parallel_fast(self, output, input, cal_total = False):
#     #     ti.static_assert(next_pow2(input.shape[0])==input.shape[0], "parallel_fast requires input count = 2**p")
#     #     n:ti.i32 = input.shape[0]
#     #     d = n >> 1
#     #     self._copy(n, output,input)
#     #     offset = 1
#     #     while(d > 0):
#     #         self._down(d,n,offset,output)
#     #         offset <<= 1
#     #         d >>= 1
        
#     #     output[n-1] = 0
#     #     d = 1
#     #     while(d < n):
#     #         offset >>= 1
#     #         self._up(d,n,offset,output)
#     #         d <<= 1
#     #     if(cal_total): return output[n-1] + input[n -1]
    
#     def exclusive_scan(self,output,input):
        # n:ti.i32 = input.shape[0]
        # npad = next_pow2(n)
        # self._resize_temp(npad)
        # self._copy_and_clear(n,npad,self.temp,input)
        # d = npad >> 1
        # offset = 1
        # while(d > 0):
        #     self._down(d,npad,offset,self.temp)
        #     offset <<= 1
        #     d >>= 1
        
        # self.temp[npad-1] = 0
        # d = 1
        # while(d < npad):
        #     offset >>= 1
        #     self._up(d,npad,offset,self.temp)
        #     d <<= 1
        # self._copy(n, output, self.temp)
        # return output[n-1] + input[n -1]


@ti.data_oriented
class BPCD:
    '''
    Broad Phase Collision Detection
    main API:
        create()
        detect_collision()
    '''
    IGNORE_USER_DATA = -1
    ExplicitCollisionPair = 1
    Implicit = 0
    @ti.dataclass
    class HashCell:
        offset : Integer
        count : Integer
        current : Integer

    def __init__(self, particle_count:Integer,hash_table_size:Integer, max_radius:Real, domain_min:Vector3, type):
        self.type = type
        self.cell_size = max_radius * 4
        self.domain_min = domain_min
        self.hash_table = BPCD.HashCell.field(shape=hash_table_size)
        self.particle_id = ti.field(Integer, particle_count)
        # collision pair list
        self.cp_list:ti.StructField
        # collision pair range
        self.cp_range:ti.StructField 
        # manage cp_list
        self.cp_tree_node:ti.SNode = None
        self.pse = PrefixSumExecutor()
        self.statistics:DEMSolverStatistics = None
        if(type == BPCD.ExplicitCollisionPair):
            self._resize_cp_list(set_collision_pair_init_capacity_factor * particle_count)
            self.cp_range = Range.field(shape=particle_count)
        
    def create(particle_count:Integer, max_radius:Real, domain_min:Vector3, domain_max:Vector3, type = Implicit):
        v = (domain_max - domain_min) / (4 * max_radius)
        size : ti.i32 = int(v[0] * v[1] * v[2])
        size = next_pow2(size)
        size = max(size, 1 << 20)
        size = min(size, 1 << 22)
        return BPCD(particle_count,size,max_radius,domain_min, type)
    
    def detect_collision(self, 
                          positions,
                          collision_resolve_callback = None):
        '''
        ### Parameters
        `positions`: field of Vector3
        `bounding_sphere_radius`: field of Real
        `collision_resolve_callback`: `func(i:ti.i32, j:ti.i32) -> None`
        ## Usage:
        ### implicit: 
            @ti.func
            
            def collision_resolve_callback(i:ti.i32, j:ti.i32):
            
                do_something_for_objecti_and_object_j()
                
                return
                
            bpcd.detect_collision(positions, collision_resolve_callback)
        
        ### explicit:
            @ti.kernel
            
            def cp_list_resolve_collision(cp_range:ti.template(), cp_list:ti.template()):
            
                for i in cp_range:
                
                    for k in range(cp_range[i].count):
                    
                        j = cp_list[cp_range[i].offset + k]
                        
                        do_something_for_objecti_and_object_j()
            
            
            bpcd.detect_collision(positions, None)
            
            cp_list = bpcd.get_collision_pair_list()
            
            cp_range = bpcd.get_collision_pair_range()
            
            cp_list_resolve_collision(cp_range, cp_list)
        '''
        if(self.type != BPCD.ExplicitCollisionPair):
            raise("In Taichi Hackathon 2022, we only support explicit collision pair")
        
        if(self.statistics!=None):self.statistics.HashTableSetupTime.tick()
        self._setup_collision(positions)
        if(self.statistics!=None):self.statistics.HashTableSetupTime.tick()

        if(self.statistics!=None):self.statistics.PrefixSumTime.tick()
        # self.pse.parallel_fast(self.hash_table.offset, self.hash_table.count)
        self.pse.exclusive_scan(self.hash_table.offset, self.hash_table.count)

        if(self.statistics!=None):self.statistics.PrefixSumTime.tick()
        
        self._put_particles(positions)
        
        if(self.statistics!=None):self.statistics.CollisionPairSetupTime.tick()
        if(collision_resolve_callback != None):
            self._solve_collision(positions, collision_resolve_callback)
        else:
            self._clear_collision_pair()
            self._search_hashtable0(positions, self.cp_list)
            total = self.pse.exclusive_scan(self.cp_range.offset, self.cp_range.count)
            if(total > self.cp_list.shape[0]):
                count = max(total, self.cp_list.shape[0] + positions.shape[0] * set_collision_pair_init_capacity_factor)
                self._resize_cp_list(count)
            self._search_hashtable1(positions, self.cp_list)
            
        if(self.statistics!=None):self.statistics.CollisionPairSetupTime.tick()

    def get_collision_pair_list(self):
        return self.cp_list
    
    def get_collision_pair_range(self):
        return self.cp_range
    
    def _resize_cp_list(self, n):
        print(f"resize_cp_list:{n}")
        ti.sync()
        # if(self.cp_tree_node!=None):
        #     self.cp_tree_node.destroy()
        fb = ti.FieldsBuilder()
        self.cp_list = ti.field(Integer)
        fb.dense(ti.i, n).place(self.cp_list)
        self.cp_tree_node = fb.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
        
    @ti.func
    def _count_particles(self, position:Vector3):
        ht = ti.static(self.hash_table)
        count = ti.atomic_add(ht[self.hash_codef(position)].count, 1)
    
    @ti.kernel
    def _put_particles(self, positions:ti.template()):
        ht = ti.static(self.hash_table)
        pid = ti.static(self.particle_id)
        for i in positions:
            hash_cell = self.hash_codef(positions[i])
            loc = ti.atomic_add(ht[hash_cell].current, 1)
            offset = ht[hash_cell].offset
            pid[offset + loc] = i

    @ti.func
    def _clear_hash_cell(self, i:Integer):
        ht = ti.static(self.hash_table)
        ht[i].offset = 0
        ht[i].current = 0
        ht[i].count = 0

    @ti.kernel
    def _search_hashtable0(self,positions:ti.template(), cp_list:ti.template()):
        cp_range = ti.static(self.cp_range)
        ht = ti.static(self.hash_table)
        for i in positions:
            o = positions[i]
            ijk = self.cell(o)
            xyz = self.cell_center(ijk)
            Zero = Vector3i(0,0,0)
            dxyz = Zero

            for k in ti.static(range(3)):
                d = o[k] - xyz[k]
                if(d > 0): dxyz[k] = 1
                else: dxyz[k] = -1

            cells = [ ijk,
                      ijk + Vector3i(dxyz[0],   0      ,    0), 
                      ijk + Vector3i(0,         dxyz[1],    0), 
                      ijk + Vector3i(0,         0,          dxyz[2]),
                      
                      ijk + Vector3i(0,         dxyz[1],    dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   0,          dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   dxyz[1],    0), 
                      ijk + dxyz 
                    ]
            
            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                if(hash_cell.count > 0):
                    for idx in range(hash_cell.offset, hash_cell.offset + hash_cell.count):
                        pid = self.particle_id[idx]
                        if(pid > i): 
                            ti.atomic_add(cp_range[i].count, 1)
    
    @ti.kernel
    def _search_hashtable1(self,positions:ti.template(), cp_list:ti.template()):
        cp_range = ti.static(self.cp_range)
        ht = ti.static(self.hash_table)
        for i in positions:
            o = positions[i]
            ijk = self.cell(o)
            xyz = self.cell_center(ijk)
            Zero = Vector3i(0,0,0)
            dxyz = Zero

            for k in ti.static(range(3)):
                d = o[k] - xyz[k]
                if(d > 0): dxyz[k] = 1
                else: dxyz[k] = -1

            cells = [ ijk,
                      ijk + Vector3i(dxyz[0],   0      ,    0), 
                      ijk + Vector3i(0,         dxyz[1],    0), 
                      ijk + Vector3i(0,         0,          dxyz[2]),
                      
                      ijk + Vector3i(0,         dxyz[1],    dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   0,          dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   dxyz[1],    0), 
                      ijk + dxyz 
                    ]
            
            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                # if(hash_cell.offset + hash_cell.count > self.particle_id.shape[0]): print(f"i = {i}, cell={self.hash_code(cells[k])}, offset = {hash_cell.offset}, count={hash_cell.count}")
                if(hash_cell.count > 0):
                    for idx in range(hash_cell.offset, hash_cell.offset + hash_cell.count):
                        pid = self.particle_id[idx]
                        if(pid > i): 
                            current = ti.atomic_add(cp_range[i].current, 1)
                            cp_list[cp_range[i].offset + current] = pid

    @ti.kernel
    def _clear_collision_pair(self):
        for i in self.cp_range:
            self.cp_range[i].offset = 0
            self.cp_range[i].count = 0
            self.cp_range[i].current = 0
    
    
    @ti.kernel
    def _setup_collision(self, positions:ti.template()):
        ht = ti.static(self.hash_table)
        # self.collision_count.fill(0)
        for i in ht: 
            self._clear_hash_cell(i)
        for i in positions: 
            self._count_particles(positions[i])
        # for i in ht: 
        #     self._fill_hash_cell(i)
    
    @ti.kernel
    def _solve_collision(self, 
                          positions:ti.template(),
                          collision_resolve_callback:ti.template()):
        ht = ti.static(self.hash_table)
        # radius = ti.static(bounding_sphere_radius)
        for i in positions:
            o = positions[i]
            # r = radius[i]
            ijk = self.cell(o)
            xyz = self.cell_center(ijk)
            Zero = Vector3i(0,0,0)
            dxyz = Zero

            for k in ti.static(range(3)):
                d = o[k] - xyz[k]
                if(d > 0): dxyz[k] = 1
                else: dxyz[k] = -1

            cells = [ ijk,
                      ijk + Vector3i(dxyz[0],   0      ,    0), 
                      ijk + Vector3i(0,         dxyz[1],    0), 
                      ijk + Vector3i(0,         0,          dxyz[2]),
                      
                      ijk + Vector3i(0,         dxyz[1],    dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   0,          dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   dxyz[1],    0), 
                      ijk + dxyz 
                    ]
            
            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                # if(hash_cell.offset + hash_cell.count > self.particle_id.shape[0]): print(f"i = {i}, cell={self.hash_code(cells[k])}, offset = {hash_cell.offset}, count={hash_cell.count}")
                if(hash_cell.count > 0):
                    for idx in range(hash_cell.offset, hash_cell.offset + hash_cell.count):
                        pid = self.particle_id[idx]
                        # other_o = positions[pid]
                        # other_r = radius[pid]
                        if(pid > i 
                        # and tm.distance(o,other_o) <= r + other_r
                        ): 
                            collision_resolve_callback(i, pid)


    @ti.kernel
    def brute_detect_collision(self,
                                positions:ti.template(), 
                                collision_resolve_callback:ti.template()):
        '''
        <Not support in Taichi Hackathon 2022>
        positions: field of Vector3
        bounding_sphere_radius: field of Real
        collision_resolve_callback: func(i:ti.i32, j:ti.i32) -> None
        '''
        for i in range(positions.shape[0]):
            # o = positions[i]
            # r = bounding_sphere_radius[i]
            for j in range(i+1, positions.shape[0]):
                # other_o = positions[j]
                # other_r = bounding_sphere_radius[j]
                # if(tm.distance(o,other_o) <= r + other_r):
                collision_resolve_callback(i, j)

    
    @ti.kernel
    def wall_detect_collision(self,
                              positions:ti.template(), 
                              collision_resolve_callback:ti.template()):
        '''
        Taichi Hackathon 2022 append
        '''
        j = 0
        for i in range(positions.shape[0]):
            collision_resolve_callback(i, j)

    # https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    @ti.func
    def morton3d32(x:Integer,y:Integer,z:Integer) -> Integer:
        answer = 0
        x &= 0x3ff
        x = (x | x << 16) & 0x30000ff
        x = (x | x << 8) & 0x300f00f
        x = (x | x << 4) & 0x30c30c3
        x = (x | x << 2) & 0x9249249
        y &= 0x3ff
        y = (y | y << 16) & 0x30000ff
        y = (y | y << 8) & 0x300f00f
        y = (y | y << 4) & 0x30c30c3
        y = (y | y << 2) & 0x9249249
        z &= 0x3ff
        z = (z | z << 16) & 0x30000ff
        z = (z | z << 8) & 0x300f00f
        z = (z | z << 4) & 0x30c30c3
        z = (z | z << 2) & 0x9249249
        answer |= x | y << 1 | z << 2
        return answer
    
    @ti.func
    def hash_codef(self, xyz:Vector3): 
        return self.hash_code(self.cell(xyz))
    
    @ti.func
    def hash_code(self, ijk:Vector3i): 
        return BPCD.morton3d32(ijk[0],ijk[1],ijk[2]) % self.hash_table.shape[0]

    @ti.func
    def cell(self, xyz:Vector3):
        ijk = ti.floor((xyz - self.domain_min) / self.cell_size, Integer)
        return ijk

    @ti.func
    def coord(self, ijk:Vector3i):
        return ijk * self.cell_size + self.domain_min

    @ti.func
    def cell_center(self, ijk:Vector3i):
        ret = Vector3(0,0,0)
        for i in ti.static(range(3)):
            ret[i] = (ijk[i] + 0.5) * self.cell_size + self.domain_min[i]
        return ret