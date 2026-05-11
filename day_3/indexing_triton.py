"""
Day 3 – Thread / Block Indexing in Triton

We mirror the apartment-complex analogy from `indexing.cu` using Triton:

- Each Triton program instance = one "apartment" (one CUDA block).
- Inside the program, a 1D vector of lanes represents the "people" (threads) in that apartment.
- We reconstruct:
  - Building (blockIdx.z)
  - Floor    (blockIdx.y)
  - Apartment(blockIdx.x)
  - Local room coordinates (threadIdx.x/y/z)
  - Linear block id and global id

Triton lacks device-side printf, so we write the metadata into a tensor and print from the host.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def indexing_kernel(
    info_ptr,  # int32[total_people, 8] flattened
    TOTAL_PEOPLE: tl.constexpr,
    APTS_PER_FLOOR: tl.constexpr,
    FLOORS: tl.constexpr,
    BUILDINGS: tl.constexpr,
    ROOM_WIDTH: tl.constexpr,
    ROOM_LENGTH: tl.constexpr,
    ROOM_HEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each "apartment" (CUDA block), compute all per-thread indices and store:

    Columns of info[row, :]
      0: global_id
      1: building (blockIdx.z)
      2: floor    (blockIdx.y)
      3: apartment(blockIdx.x)
      4: room_x   (threadIdx.x)
      5: room_y   (threadIdx.y)
      6: room_z   (threadIdx.z)
      7: block_id (linear block id)
    """
    pid = tl.program_id(axis=0)  # this is our "block_id"

    # Decode blockIdx.{x,y,z} from linear block_id
    grid_x = APTS_PER_FLOOR
    grid_y = FLOORS
    # grid_z = BUILDINGS

    apartment = pid % grid_x
    tmp = pid // grid_x
    floor = tmp % grid_y
    building = tmp // grid_y

    people_per_apartment = BLOCK_SIZE
    base = pid * people_per_apartment  # starting global_id for this apartment

    # Local thread ids within apartment: 0..BLOCK_SIZE-1
    local = tl.arange(0, BLOCK_SIZE)

    # Decode room coordinates from local_thread_id
    room_x = local % ROOM_WIDTH
    room_y = (local // ROOM_WIDTH) % ROOM_LENGTH
    room_z = local // (ROOM_WIDTH * ROOM_LENGTH)

    global_id = base + local

    # Bounds mask (should be all true if TOTAL_PEOPLE == num_apartments * BLOCK_SIZE)
    mask = global_id < TOTAL_PEOPLE

    # info is laid out as [TOTAL_PEOPLE, 8] with row_stride = 8
    row_stride = 8
    # Write each column
    tl.store(info_ptr + global_id * row_stride + 0, global_id.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 1, building.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 2, floor.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 3, apartment.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 4, room_x.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 5, room_y.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 6, room_z.to(tl.int32), mask=mask)
    tl.store(info_ptr + global_id * row_stride + 7, pid.to(tl.int32), mask=mask)


def run_indexing_demo_triton() -> None:
    # Match the configuration from `indexing.cu`
    BUILDINGS = 2        # gridDim.z
    FLOORS = 2           # gridDim.y
    APTS_PER_FLOOR = 2   # gridDim.x

    ROOM_WIDTH = 4       # blockDim.x
    ROOM_LENGTH = 2      # blockDim.y
    ROOM_HEIGHT = 2      # blockDim.z

    total_apartments = BUILDINGS * FLOORS * APTS_PER_FLOOR
    people_per_apartment = ROOM_WIDTH * ROOM_LENGTH * ROOM_HEIGHT
    total_people = total_apartments * people_per_apartment

    print("Triton Apartment Complex Configuration:")
    print("--------------------------------------")
    print(
        f"Buildings: {BUILDINGS}, Floors per Building: {FLOORS}, "
        f"Apartments per Floor: {APTS_PER_FLOOR}"
    )
    print(
        f"Room Dimensions (W x L x H): "
        f"{ROOM_WIDTH} x {ROOM_LENGTH} x {ROOM_HEIGHT}"
    )
    print(f"Total Apartments: {total_apartments}")
    print(f"People per Apartment: {people_per_apartment}")
    print(f"Total People: {total_people}")
    print("--------------------------------------\n")

    if not torch.cuda.is_available():
        print("CUDA not available; Triton demo requires a CUDA GPU.")
        return

    device = torch.device("cuda")

    # Allocate info tensor: [total_people, 8] int32
    info = torch.empty((total_people, 8), device=device, dtype=torch.int32)

    # Launch one program per apartment
    grid = (total_apartments,)
    indexing_kernel[grid](
        info,
        TOTAL_PEOPLE=total_people,
        APTS_PER_FLOOR=APTS_PER_FLOOR,
        FLOORS=FLOORS,
        BUILDINGS=BUILDINGS,
        ROOM_WIDTH=ROOM_WIDTH,
        ROOM_LENGTH=ROOM_LENGTH,
        ROOM_HEIGHT=ROOM_HEIGHT,
        BLOCK_SIZE=people_per_apartment,
    )

    # Bring results back to host and print in a readable form
    info_host = info.cpu().numpy()
    for row in info_host:
        global_id, building, floor, apartment, room_x, room_y, room_z, block_id = row.tolist()
        print(
            f"Person {global_id} | Building-Floor-Apt({building},{floor},{apartment}) | "
            f"Room-Position({room_x},{room_y},{room_z}) | "
            f"Apartment#{block_id} | Room#{global_id - block_id * people_per_apartment}"
        )


if __name__ == "__main__":
    run_indexing_demo_triton()

