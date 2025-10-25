#include <stdio.h>
#include <string.h>
#include "esp_system.h"
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_vfs_fat.h"
#include "ff.h"
#include <inttypes.h>
#include "sdmmc_cmd.h"


static const char *TAG = "AUDIO_PLAYER";
static const gpio_num_t led_pin = GPIO_NUM_4;

// WAV format constants
#define SAMPLE_RATE     44100
#define BITS_PER_SAMPLE 16
#define CHANNELS        2

// SPI Pins defined previously
#define PIN_NUM_MISO  13
#define PIN_NUM_MOSI  11
#define PIN_NUM_CLK   12
#define PIN_NUM_CS    10

// I2S Pins defined previously
#define I2S_BCK_PIN   8
#define I2S_WS_PIN    7
#define I2S_DATA_PIN  9

// WAV header struct (44 bytes)
typedef struct {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_header[4];
    uint32_t data_size;
} wav_header_t;

static i2s_chan_handle_t tx_chan;

static void init_i2s(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &tx_chan, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_16BIT,
                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCK_PIN,
            .ws = I2S_WS_PIN,
            .dout = I2S_DATA_PIN,
            .din = I2S_GPIO_UNUSED
        }
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(tx_chan));
}

static void init_sdcard(void) {
    ESP_LOGI(TAG, "Initializing SD card (SPI mode)");

    // Install ISR service if not installed yet
    esp_err_t ret = gpio_install_isr_service(0);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "GPIO ISR install failed: %s", esp_err_to_name(ret));
    }

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000
    };

    ESP_LOGI(TAG, "Initializing SPI bus");
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_cfg, SDSPI_DEFAULT_DMA));

    sdspi_device_config_t slot_cfg = {
        .host_id = SPI2_HOST,
        .gpio_cs = PIN_NUM_CS,
        .gpio_cd = SDSPI_SLOT_NO_CD,
        .gpio_wp = SDSPI_SLOT_NO_WP,
    };

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 4,
    };

    sdmmc_card_t *card;
    ESP_LOGI(TAG, "Mounting filesystem");
    ESP_ERROR_CHECK(esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_cfg, &mount_cfg, &card));

    ESP_LOGI(TAG, "SD card mounted successfully");
    sdmmc_card_print_info(stdout, card);
}

static void play_wav_file(const char *path) {
    ESP_LOGI(TAG, "Opening WAV file: %s", path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "File not found");
        return;
    }

    wav_header_t header;
    fread(&header, sizeof(header), 1, f);

    ESP_LOGI(TAG, "Sample rate: %" PRIu32, header.sample_rate);
    ESP_LOGI(TAG, "Channels: %" PRIu16, header.num_channels);
    ESP_LOGI(TAG, "Bits per sample: %" PRIu16, header.bits_per_sample);


    uint8_t buffer[512];
    size_t bytes_read;
    size_t bytes_written;

    while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0) {
        i2s_channel_write(tx_chan, buffer, bytes_read, &bytes_written, portMAX_DELAY);
    }

    fclose(f);
    ESP_LOGI(TAG, "Playback complete");
}

void app_main(void) {

    ESP_LOGI(TAG, "Starting SD card bring-up test");
    gpio_reset_pin(led_pin);
    gpio_set_direction(led_pin, GPIO_MODE_OUTPUT);
    
    init_sdcard();
    init_i2s();

    play_wav_file("/sdcard/001.wav");


    // FILE *f = fopen("/sdcard/001.wav", "rb");
    // if (!f) {
    //     ESP_LOGI(TAG, "Failed to open /sdcard/001.wav");
    //     return;
    // }
    // else
    // {
    //     gpio_set_level(led_pin, 1);
    // }

    // ESP_LOGI(TAG, "WAV file opened OK");

    // wav_header_t header;
    // fread(&header, sizeof(header), 1, f);
    // ESP_LOGI(TAG, "Sample Rate: %" PRIu32, header.sample_rate);
    // ESP_LOGI(TAG, "Channels: %" PRIu16, header.num_channels);
    // ESP_LOGI(TAG, "Bits per sample: %" PRIu16, header.bits_per_sample);

    // fclose(f);
    // ESP_LOGI(TAG, "SD access OK");
}