import taichi as ti

WARP_SZ = 32
BLOCK_SZ = 128

@ti.data_oriented
class PrefixSumExecutor:
# target = ti.cfg.cuda
    def __init__(self):
        self.smem = None
        self.arrs = []
        self.ele_nums = []
        self.ele_nums_pos = []
        self.large_arr = None
        self.grid_size = 0
        self.large_arr_size = 0
        self.last_length = 0
    
    @ti.func
    def _warp_inclusive_add(self, val: ti.template()):
        global_tid = ti.global_thread_idx()
        lane_id = global_tid % 32
        # Intra-warp scan, manually unroll
        offset_j = 1
        n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
        if (lane_id >= offset_j):
            val += n
        offset_j = 2
        n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
        if (lane_id >= offset_j):
            val += n
        offset_j = 4
        n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
        if (lane_id >= offset_j):
            val += n
        offset_j = 8
        n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
        if (lane_id >= offset_j):
            val += n
        offset_j = 16
        n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
        if (lane_id >= offset_j):
            val += n
        return val

    @ti.func
    def _barrier(self):
        ti.simt.block.sync()
    
    @ti.kernel
    def _shfl_scan(self, arr_in: ti.template(), in_beg: ti.i32, in_end: ti.i32,
                sum_smem: ti.template(), single_block: ti.template()):
        ti.loop_config(block_dim=BLOCK_SZ)
        for i in range(in_beg, in_end):
            val = arr_in[i]

            thread_id = i % BLOCK_SZ
            block_id = int((i - in_beg) // BLOCK_SZ)
            lane_id = thread_id % WARP_SZ
            warp_id = thread_id // WARP_SZ

            val = self._warp_inclusive_add(val)
            self._barrier()

            # Put warp scan results to smem
            if (thread_id % WARP_SZ == WARP_SZ - 1):
                sum_smem[block_id, warp_id] = val
            self._barrier()

            # Inter-warp scan, use the first thread in the first warp
            if (warp_id == 0 and lane_id == 0):
                for k in range(1, BLOCK_SZ // WARP_SZ):
                    sum_smem[block_id, k] += sum_smem[block_id, k - 1]
            self._barrier()

            # Update data with warp sums
            warp_sum = 0
            if (warp_id > 0):
                warp_sum = sum_smem[block_id, warp_id - 1]
            val += warp_sum
            arr_in[i] = val

            # Update partial sums
            if not single_block and (thread_id == BLOCK_SZ - 1):
                arr_in[in_end + block_id] = val


    @ti.kernel
    def _uniform_add(self, arr_in: ti.template(), in_beg: ti.i32, in_end: ti.i32):
        ti.loop_config(block_dim=BLOCK_SZ)
        for i in range(in_beg + BLOCK_SZ, in_end):
            block_id = int((i - in_beg) // BLOCK_SZ)
            arr_in[i] += arr_in[in_end + block_id - 1]


    @ti.kernel
    def _blit_from_field_to_field(
        self,
        dst: ti.template(), src: ti.template(), offset: ti.i32, size: ti.i32):
        for i in range(size): dst[i + offset] = src[i]
    
    @ti.kernel
    def _blit_from_field_to_field_shift_one(
        self,
        dst: ti.template(), src: ti.template(), size: ti.i32):
        for i in range(size): 
            if(i == 0): dst[i] = 0
            else: dst[i] = src[i-1]

    def _parallel_prefix_sum_inclusive_in_private_array(self, input_arr, length):
        if(self.last_length != length):
            grid_size = int((length + BLOCK_SZ - 1) // BLOCK_SZ)
            # Declare input array and all partial sums
            ele_num = length
            # Get starting position and length
            self.ele_nums.clear()
            self.ele_nums_pos.clear()
            self.ele_nums.append(ele_num)
            start_pos = 0
            self.ele_nums_pos.append(start_pos)
            
            while (ele_num > 1):
                ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
                self.ele_nums.append(ele_num)
                start_pos += BLOCK_SZ * ele_num
                self.ele_nums_pos.append(start_pos)
            
            large_arr_size = start_pos
            if(self.grid_size < grid_size):
                self.smem = ti.field(ti.i32, shape=(int(grid_size), 64))
                self.grid_size = grid_size
                print(f"prefix sum smem resize:{grid_size}x64")
            if(self.large_arr_size < large_arr_size):
                self.large_arr = ti.field(ti.i32, shape = large_arr_size)
                self.large_arr_size = large_arr_size
                print(f"prefix sum temp resize:{large_arr_size}")
            self.last_length = length

        self._blit_from_field_to_field(self.large_arr, input_arr, 0, length)

        for i in range(len(self.ele_nums) - 1):
            if i == len(self.ele_nums) - 2:
                self._shfl_scan(self.large_arr, self.ele_nums_pos[i], self.ele_nums_pos[i + 1], self.smem, True)
            else:
                self._shfl_scan(self.large_arr, self.ele_nums_pos[i], self.ele_nums_pos[i + 1], self.smem, False)

        for i in range(len(self.ele_nums) - 3, -1, -1):
            self._uniform_add(self.large_arr, self.ele_nums_pos[i], self.ele_nums_pos[i + 1])
            
    def inclusive_scan_inplace(self, input_arr, length = -1):
        if(length < 0): length = input_arr.shape[0]
        self._parallel_prefix_sum_inclusive_in_private_array(input_arr, length)
        self._blit_from_field_to_field(input_arr, self.large_arr, 0, length)
        
    def inclusive_scan(self, output_arr, input_arr, length = -1):
        if(length < 0): length = input_arr.shape[0]
        self._parallel_prefix_sum_inclusive_in_private_array(input_arr, length)
        self._blit_from_field_to_field(output_arr, self.large_arr, 0, length)
    
    def exclusive_scan_inplace(self, input_arr, length = -1):
        if(length < 0): length = input_arr.shape[0]
        self._parallel_prefix_sum_inclusive_in_private_array(input_arr, length)
        self._blit_from_field_to_field_shift_one(input_arr, self.large_arr, length)
        return self.large_arr[length-1]
    
    def exclusive_scan(self,output_arr,input_arr, length = -1):
        if(length < 0): length = input_arr.shape[0]
        self._parallel_prefix_sum_inclusive_in_private_array(input_arr, length)
        self._blit_from_field_to_field_shift_one(output_arr, self.large_arr, length)
        return self.large_arr[length-1]

    
if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    executor = PrefixSumExecutor()
    arr = ti.field(ti.i32, shape=256)
    arr.fill(1)
    sum1 = executor.exclusive_scan_inplace(arr, 64)
    print(arr.to_numpy())
    arr.fill(1)
    sum2 = executor.exclusive_scan_inplace(arr, 256)
    print(arr.to_numpy())