#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "memory.h"

/* Pointer to simulator memory */
uint8_t *mem;

/* Called by program loader to initialize memory. */
uint8_t *init_mem() {
  assert (mem == NULL);
  mem = calloc(MEM_SIZE, sizeof(uint8_t)); // allocate zeroed memory
  return mem;
}

/* Returns 1 if memory access is ok, otherwise 0 */
int access_ok(uint32_t mipsaddr, mem_unit_t size) {
    
    if(mipsaddr > 0 && mipsaddr < MEM_SIZE)
    {
        switch (size)
        {
            case SIZE_WORD:
                if(mipsaddr % 4 == 0) /* Must be divisible by 4 */
                {
                    return 1;
                }
                break;
                
            case SIZE_HALF_WORD: /* Must be divisible by 2 */
                if(mipsaddr % 2 == 0)
                {
                    return 1;
                }
                break;
                
            case SIZE_BYTE: /* Always properly aligned */
                return 1;
                break;
        }
    }

    return 0;
}

/* Writes size bytes of value into mips memory at mipsaddr */
void store_mem(uint32_t mipsaddr, mem_unit_t size, uint32_t value) {
    
    int i;
    
    if (!access_ok(mipsaddr, size)) {
        fprintf(stderr, "%s: bad write=%08x\n", __FUNCTION__, mipsaddr);
        exit(-1);
    }

    switch(size)
    {
        case SIZE_WORD:
            for(i = 0; i < SIZE_WORD; i++)
                *(mem + mipsaddr + i) = (value & (0x000000FF << 8*i)) >> 8*i;
            break;
            
        case SIZE_HALF_WORD:
            for(i = 0; i < SIZE_HALF_WORD; i++)
                *(mem + mipsaddr + i) = (value & (0x00FF << 8*i)) >> 8*i;
            break;
            
        case SIZE_BYTE:
            *(mem + mipsaddr) = value;
            break;
    }

}

/* Returns zero-extended value from mips memory */
uint32_t load_mem(uint32_t mipsaddr, mem_unit_t size) {
    if (!access_ok(mipsaddr, size)) {
        fprintf(stderr, "%s: bad read=%08x\n", __FUNCTION__, mipsaddr);
        exit(-1);
    }
    
    switch(size)
    {
        case SIZE_BYTE:
            return *(uint8_t*)(mem + mipsaddr);
            break;
            
        case SIZE_HALF_WORD:
            return *(uint16_t*)(mem + mipsaddr);
            break;
        
        case SIZE_WORD:
            break;
    }

  // incomplete stub to let mipscode/simple execute
  // (only handles size == SIZE_WORD correctly)
  // feel free to delete and implement your own way
  return *(uint32_t*)(mem + mipsaddr);
}
