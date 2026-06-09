#include "utils.h"

/* Converts 8.24 format from SigmaStudio to float e.g. to readback parameter values from DSP */
float convert_DSP_to_float_(uint32_t sigmastudio_value){
    
    return (float)((int32_t)sigmastudio_value / 16777216.0f);


}

/*Converts float to 8.24 DSP-readable format e.g. when writing new parameters over SPI*/

uint32_t convert_float_to_DSP_(float val){
    return (uint32_t)(val * 16777216.0f);
}


/*Repackages an 8-bit buffer into a 32 bit word e.g. when handling rx buffers from SPI */
uint32_t repackage_8_bit_to_32_bit_(uint8_t *data){

    return (uint32_t)data[0] << 24 | (uint32_t)data[1] << 16 | (uint32_t)data[2] << 8 | (uint32_t)data[3];
    
}


/*Repackages a 32-bit word into a uint8_t array e.g. to tx new parameters over SPI */
void repackage_32_bit_to_8_bit_(uint8_t *data, uint32_t oldVal){

    data[0] = (uint8_t)(oldVal >> 24);
    data[1] = (uint8_t)(oldVal >> 16);
    data[2] = (uint8_t)(oldVal >> 8);
    data[3] = (uint8_t)(oldVal); 


}