#include "ADAU1467_SPI.h"
#include "mxc.h"

#include <stdio.h>
#include <string.h>

static int spi_initialized = 0;

#define ADAU1467_CS_PORT    MXC_GPIO0
#define ADAU1467_CS_PIN     MXC_GPIO_PIN_11

static int cs_gpio_initialized = 0;

static void cs_init(void)
{
    if (!cs_gpio_initialized) {
        mxc_gpio_cfg_t cs_cfg;
        cs_cfg.port  = ADAU1467_CS_PORT;
        cs_cfg.mask  = ADAU1467_CS_PIN;
        cs_cfg.func  = MXC_GPIO_FUNC_OUT;
        cs_cfg.pad   = MXC_GPIO_PAD_NONE;
        cs_cfg.vssel = MXC_GPIO_VSSEL_VDDIOH;
        MXC_GPIO_Config(&cs_cfg);
        MXC_GPIO_OutSet(ADAU1467_CS_PORT, ADAU1467_CS_PIN);
        cs_gpio_initialized = 1;
    }
}

static void toggle_ss(void)
{
    cs_init();
    MXC_GPIO_OutClr(ADAU1467_CS_PORT, ADAU1467_CS_PIN);
    MXC_Delay(MXC_DELAY_USEC(10));
    MXC_GPIO_OutSet(ADAU1467_CS_PORT, ADAU1467_CS_PIN);
    MXC_Delay(MXC_DELAY_USEC(10));
}

int adau1467_spi_init(void)
{
    int ret;

    /* Deselect onboard QSPI SRAM (P0.10) and microSD (P0.4) */
    mxc_gpio_cfg_t cs_cfg;
    cs_cfg.func  = MXC_GPIO_FUNC_OUT;
    cs_cfg.pad   = MXC_GPIO_PAD_NONE;
    cs_cfg.vssel = MXC_GPIO_VSSEL_VDDIOH;

    cs_cfg.port = MXC_GPIO0;
    cs_cfg.mask = MXC_GPIO_PIN_10;
    MXC_GPIO_Config(&cs_cfg);
    MXC_GPIO_OutSet(MXC_GPIO0, MXC_GPIO_PIN_10);

    cs_cfg.mask = MXC_GPIO_PIN_4;
    MXC_GPIO_Config(&cs_cfg);
    MXC_GPIO_OutSet(MXC_GPIO0, MXC_GPIO_PIN_4);

    /* Toggle SS on P0.11 (GPIO) to switch ADAU1467 from I2C to SPI mode */
    printf("ADAU1467: Toggling SS to switch to SPI mode...\r\n");
    for (int i = 0; i < 3; i++) {
        toggle_ss();
    }
    MXC_Delay(MXC_DELAY_MSEC(10));

    /* Now init SPI with SS1 on P0.11 (hardware-managed) */
    mxc_spi_pins_t spi_pins;
    spi_pins.clock = TRUE;
    spi_pins.miso  = TRUE;
    spi_pins.mosi  = TRUE;
    spi_pins.sdio2 = FALSE;
    spi_pins.sdio3 = FALSE;
    spi_pins.ss0   = FALSE;    /* P0.10 = SRAM, do not use */
    spi_pins.ss1   = TRUE;     /* P0.11 = ADAU1467 CS */
    spi_pins.ss2   = FALSE;

    ret = MXC_SPI_Init(ADAU1467_SPI_INSTANCE, 1, 0, 1, 0,
                        ADAU1467_SPI_SPEED_HZ, spi_pins);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI init failed: %d\r\n", ret);
        return ret;
    }

    ret = MXC_SPI_SetMode(ADAU1467_SPI_INSTANCE, SPI_MODE_3);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI set mode failed: %d\r\n", ret);
        return ret;
    }

    ret = MXC_SPI_SetWidth(ADAU1467_SPI_INSTANCE, SPI_WIDTH_STANDARD);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI set width failed: %d\r\n", ret);
        return ret;
    }

    ret = MXC_SPI_SetDataSize(ADAU1467_SPI_INSTANCE, 8);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI set data size failed: %d\r\n", ret);
        return ret;
    }

    spi_initialized = 1;
    printf("ADAU1467 SPI init OK\r\n");
    return 0;
}


int adau1467_spi_write(uint16_t addr, const uint8_t *data, size_t len)
{
    if (!spi_initialized) return -1;

    size_t total_len = 3 + len;
    uint8_t tx_buf[64];

    if (total_len > sizeof(tx_buf)) {
        printf("ADAU1467 SPI write: too large (%u bytes)\r\n", (unsigned)len);
        return -2;
    }

    tx_buf[0] = ADAU1467_SPI_CHIP_ADDR_W;
    tx_buf[1] = (uint8_t)(addr >> 8);
    tx_buf[2] = (uint8_t)(addr & 0xFF);
    memcpy(&tx_buf[3], data, len);

    mxc_spi_req_t req;
    memset(&req, 0, sizeof(req));
    req.spi        = ADAU1467_SPI_INSTANCE;
    req.txData     = tx_buf;
    req.rxData     = NULL;
    req.txLen      = total_len;
    req.rxLen      = 0;
    req.ssIdx      = ADAU1467_SPI_SS_IDX;
    req.ssDeassert = 1;
    req.completeCB = NULL;

    int ret = MXC_SPI_MasterTransaction(&req);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI write failed @ 0x%04X: %d\r\n", addr, ret);
    }
    return ret;
}


int adau1467_spi_read(uint16_t addr, uint8_t *data, size_t len)
{
    if (!spi_initialized) return -1;

    size_t total_len = 3 + len;
    uint8_t tx_buf[64];
    uint8_t rx_buf[64];

    if (total_len > sizeof(tx_buf)) return -2;

    memset(tx_buf, 0, total_len);
    memset(rx_buf, 0, total_len);

    tx_buf[0] = ADAU1467_SPI_CHIP_ADDR_R;
    tx_buf[1] = (uint8_t)(addr >> 8);
    tx_buf[2] = (uint8_t)(addr & 0xFF);

    mxc_spi_req_t req;
    memset(&req, 0, sizeof(req));
    req.spi        = ADAU1467_SPI_INSTANCE;
    req.txData     = tx_buf;
    req.rxData     = rx_buf;
    req.txLen      = total_len;
    req.rxLen      = total_len;
    req.ssIdx      = ADAU1467_SPI_SS_IDX;
    req.ssDeassert = 1;
    req.completeCB = NULL;

    int ret = MXC_SPI_MasterTransaction(&req);
    if (ret != E_NO_ERROR) {
        printf("ADAU1467 SPI read failed @ 0x%04X: %d\r\n", addr, ret);
        return ret;
    }

    memcpy(data, &rx_buf[3], len);
    return 0;
}


int adau1467_safeload_write(uint16_t param_addr, uint32_t value)
{
    int ret;

    /* Step 1: Write data to safeload data slot 0 */
    uint8_t data_bytes[4] = {
        (uint8_t)(value >> 24),
        (uint8_t)(value >> 16),
        (uint8_t)(value >> 8),
        (uint8_t)(value)
    };
    ret = adau1467_spi_write(ADAU1467_SAFELOAD_DATA_BASE, data_bytes, 4);
    if (ret != 0) return ret;

    /* Step 2: Write target parameter RAM address */
    uint8_t addr_bytes[4] = {
        0x00,
        0x00,
        (uint8_t)(param_addr >> 8),
        (uint8_t)(param_addr & 0xFF)
    };
    ret = adau1467_spi_write(ADAU1467_SAFELOAD_ADDR, addr_bytes, 4);
    if (ret != 0) return ret;

    /* Step 3: Write word count to Lower (Page 1 / lower memory) */
    uint8_t num_lower[4] = {0x00, 0x00, 0x00, 0x01};
    ret = adau1467_spi_write(ADAU1467_SAFELOAD_NUM_LOWER, num_lower, 4);
    if (ret != 0) return ret;

    /* Step 4: Write 0 to Upper — THIS triggers the safeload */
    uint8_t num_upper[4] = {0x00, 0x00, 0x00, 0x00};
    ret = adau1467_spi_write(ADAU1467_SAFELOAD_NUM_UPPER, num_upper, 4);
    if (ret != 0) return ret;

    /* Wait one audio frame for safeload to complete.
     * At 48 kHz, one frame = 20.83 us. Use 50 us for margin. */
    MXC_Delay(MXC_DELAY_USEC(25));

    return 0;
}