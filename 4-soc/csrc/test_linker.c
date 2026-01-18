#include <stdint.h>
#include <stddef.h>

/** 
 * Include the header containing the static const cube[] array.
 * Because of my linker script, this data will be read from the 
 * .assets section (likely starting at 0x00002000) in expectation.
 */
#include "b3d-assets.h"

int main(void) {
    volatile int32_t checksum = 0;
    
    // Calculate the number of elements in the array
    size_t array_len = sizeof(cube) / sizeof(cube[0]);

    // If the linker script is correct, this reads from 0x2000 onwards.
    // (which means that you'll see he data from .assets section through 
    // the command `riscv-none-elf-objdump -Ds test_linker.elf | less`)
    for (size_t i = 0; i < array_len; i++) {
        checksum += cube[i];
    }

    // 2. Data Integrity Check
    volatile int system_status = 0; // 0 = Unknown, 1 = OK, -1 = Fail

    if (cube[0] == 0x0000000C) {
        system_status = 1;  // Success: Data loaded correctly
    } else {
        system_status = -1; // Fail: Memory map mismatch
    }
    
    while (1) {
        checksum++; 
    }

    return 0;
}