#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "mxc_delay.h"
#include "ADAU1467_SPI.h"
#include "beam_table.h"
#include "utils.h"

/* ---- Update this after re-exporting PARAM.h ---- */
#define LEVEL_READBACK_ADDR  50   /* replace with new SingleLevelDetector LEVEL addr */

/* ---- Tuning ---- */
#define SETTLE_US  250   /* start safe; reduce later if stable */

/* ---- Steering ---- */

int steer_beam(int angle_index)
{
    for (int i = 0; i < BEAM_NUM_MICS; i++) {
        int ret = adau1467_safeload_write(
            beam_delay_addrs[i],
            beam_delay_fixpt[angle_index][i]
        );
        if (ret != 0) {
            printf("Steer failed: mic %d, angle_idx %d, ret %d\r\n",
                   i, angle_index, ret);
            return ret;
        }
    }
    return 0;
}

/* ---- Single readback from level detector after RMS envelope ---- */

uint32_t read_level_raw(void)
{
    uint8_t buf[4] = {0};
    int ret = adau1467_spi_read(LEVEL_READBACK_ADDR, buf, 4);
    if (ret != 0) {
        return 0;
    }
    return repackage_8_bit_to_32_bit_(buf);
}

int main(void)
{
    printf("ADAU1467 beam scan (RMS -> Level Detector)\r\n");

    int ret = adau1467_spi_init();
    if (ret != 0) {
        printf("SPI init failed: %d\r\n", ret);
        while (1) {}
    }

    /* Verify SPI link */
    uint8_t verify[4] = {0};
    adau1467_spi_read(beam_delay_addrs[0], verify, 4);
    printf("Verify addr %d: %02X %02X %02X %02X\r\n",
           beam_delay_addrs[0], verify[0], verify[1], verify[2], verify[3]);

    /* Quick check of level register */
    uint8_t lvl[4] = {0};
    adau1467_spi_read(LEVEL_READBACK_ADDR, lvl, 4);
    printf("Level addr %d: %02X %02X %02X %02X\r\n",
           LEVEL_READBACK_ADDR, lvl[0], lvl[1], lvl[2], lvl[3]);

    while (1) {
        for (int i = 0; i < BEAM_NUM_ANGLES; i++) {
            steer_beam(i);
            MXC_Delay(MXC_DELAY_USEC(SETTLE_US));

            uint32_t level_raw = read_level_raw();

            /* Send raw integer to PC */
            printf("P,%d,%lu\r\n",
                   beam_angles_deg[i],
                   (unsigned long)level_raw);
        }

        printf("SCAN_DONE\r\n");
    }

    return 0;
}