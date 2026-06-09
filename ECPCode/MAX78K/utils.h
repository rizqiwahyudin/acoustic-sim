#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

float convert_DSP_to_float_(uint32_t sigmastudio_value); 
uint32_t convert_float_to_DSP_(float val);
uint32_t repackage_8_bit_to_32_bit_(uint8_t *data);
void repackage_32_bit_to_8_bit_(uint8_t *data, uint32_t oldVal);