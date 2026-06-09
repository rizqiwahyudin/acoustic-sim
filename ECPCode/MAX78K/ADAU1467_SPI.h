#ifndef ADAU1467_SPI_H
#define ADAU1467_SPI_H

#include <stdint.h>
#include <stddef.h>
#include "spi.h"

#define ADAU1467_SPI_INSTANCE       MXC_SPI0
#define ADAU1467_SPI_SPEED_HZ       4000000
#define ADAU1467_SPI_SS_IDX         1

#define ADAU1467_SPI_CHIP_ADDR_W    0x00
#define ADAU1467_SPI_CHIP_ADDR_R    0x01

#define ADAU1467_SAFELOAD_DATA_BASE     0x6000
#define ADAU1467_SAFELOAD_DATA_SIZE     4
#define ADAU1467_SAFELOAD_NUM_SLOTS     5

#define ADAU1467_SAFELOAD_ADDR          0x6005
#define ADAU1467_SAFELOAD_NUM_LOWER     0x6006
#define ADAU1467_SAFELOAD_NUM_UPPER     0x6007

int adau1467_spi_init(void);
int adau1467_spi_write(uint16_t addr, const uint8_t *data, size_t len);
int adau1467_spi_read(uint16_t addr, uint8_t *data, size_t len);
int adau1467_safeload_write(uint16_t param_addr, uint32_t value);

#endif /* ADAU1467_SPI_H */