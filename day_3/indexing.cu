#include <stdio.h>

/**
 * Visualizing CUDA Thread Organization as an Apartment Complex:
 * 
 * Imagine a city with multiple apartment buildings:
 * - Each building represents one z-level (blockIdx.z)
 * - Each floor in a building represents one y-level (blockIdx.y)
 * - Each apartment on a floor represents one x-position (blockIdx.x)
 * - Inside each apartment, we have people organized in a 3D room (threadIdx.x/y/z)
 */
__global__ void visualizeThreadOrganization() {
    // Step 1: Calculate which building-floor-apartment (block) we're in
    int building = blockIdx.z;    // Which building in the complex
    int floor = blockIdx.y;       // Which floor in the building
    int apartment = blockIdx.x;   // Which apartment on the floor

    // Calculate unique block ID (like a unique apartment number across all buildings)
    int block_id = apartment +                     // Position on floor
                  floor * gridDim.x +             // Offset for floor number
                  building * gridDim.x * gridDim.y; // Offset for building number

    // Step 2: Calculate position within the apartment (thread position)
    int room_x = threadIdx.x;     // Position along width of room
    int room_y = threadIdx.y;     // Position along length of room
    int room_z = threadIdx.z;     // Position along height of room

    // Calculate local position within block (like room number in apartment)
    int local_thread_id = room_x +                          // Position in room width
                         room_y * blockDim.x +              // Offset for room length
                         room_z * blockDim.x * blockDim.y;  // Offset for room height

    // Step 3: Calculate global unique ID
    // First, calculate how many people fit in one apartment
    int people_per_apartment = blockDim.x * blockDim.y * blockDim.z;
    
    // Then calculate global ID (unique number for each person in entire complex)
    int global_id = block_id * people_per_apartment + local_thread_id;

    // Print information in a clear, structured format
    printf("Person %d | Building-Floor-Apt(%d,%d,%d) | Room-Position(%d,%d,%d) | "
           "Apartment#%d | Room#%d\n",
           global_id,                              // Unique global ID
           building, floor, apartment,             // Block position
           room_x, room_y, room_z,                // Thread position
           block_id,                              // Unique block ID
           local_thread_id);                      // Local position in block
}

int main() {
    // Define our apartment complex dimensions
    const int BUILDINGS = 2;      // Number of buildings (z-direction)
    const int FLOORS = 2;         // Floors per building (y-direction)
    const int APTS_PER_FLOOR = 2; // Apartments per floor (x-direction)

    // Define dimensions within each apartment
    const int ROOM_WIDTH = 4;     // Room width (x-direction)
    const int ROOM_LENGTH = 2;    // Room length (y-direction)
    const int ROOM_HEIGHT = 2;    // Room height (z-direction)

    // Calculate total numbers for verification
    int total_apartments = BUILDINGS * FLOORS * APTS_PER_FLOOR;
    int people_per_apartment = ROOM_WIDTH * ROOM_LENGTH * ROOM_HEIGHT;
    int total_people = total_apartments * people_per_apartment;

    // Print configuration summary
    printf("Apartment Complex Configuration:\n");
    printf("--------------------------------\n");
    printf("Buildings: %d, Floors per Building: %d, Apartments per Floor: %d\n",
           BUILDINGS, FLOORS, APTS_PER_FLOOR);
    printf("Room Dimensions (W x L x H): %d x %d x %d\n",
           ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT);
    printf("Total Apartments: %d\n", total_apartments);
    printf("People per Apartment: %d\n", people_per_apartment);
    printf("Total People: %d\n", total_people);
    printf("--------------------------------\n\n");

    // Set up grid and block dimensions
    dim3 gridDim(APTS_PER_FLOOR, FLOORS, BUILDINGS);
    dim3 blockDim(ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT);

    // Launch kernel
    visualizeThreadOrganization<<<gridDim, blockDim>>>();
    
    // Wait for all threads to complete
    cudaDeviceSynchronize();
    
    // Check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}