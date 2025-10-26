#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_system.h"
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_vfs_fat.h"
#include "ff.h"
#include <inttypes.h>
#include "sdmmc_cmd.h"
#include "esp_dsp.h"

static const char *TAG = "AUDIO_PLAYER";
static const gpio_num_t led_pin = GPIO_NUM_4;

// WAV format constants
#define SAMPLE_RATE 44100
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

#define N_SAMPLES 2048
#define N_FFT_BINS N_SAMPLES/2
#define HOP N_SAMPLES/2
#define N_BINS 8

static wav_header_t wav_hdr;

// Time-domain working buffers (float)
static float x1[N_SAMPLES] __attribute__((aligned(16)));   // left/mono
static float x2[N_SAMPLES] __attribute__((aligned(16)));   // right or copy of mono

// Hann window, complex work buffer
static float wind[N_SAMPLES] __attribute__((aligned(16)));
static float y_cf[N_SAMPLES * 2] __attribute__((aligned(16)));   // interleaved complex

// After split aliases
static float *y1_cf = &y_cf[0];
static float *y2_cf = &y_cf[N_SAMPLES];

// Magnitude (linear 0..1 here)
static float mag1[N_SAMPLES/2];
static float mag2[N_SAMPLES/2];

// Combines the L and R amplitudes from the stereo sound
static float mono[N_FFT_BINS];
// Condenses the N_FFT_BINS number of freq bins into 8 
static float bins8[8];

// -------------------- Utilities --------------------
static inline bool fread_exact(void *dst, size_t n, FILE *f) {
    return fread(dst, 1, n, f) == n;
}
static inline uint32_t rd_u32le(FILE *f) {
    uint8_t b[4]; if (!fread_exact(b,4,f)) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1]<<8) | ((uint32_t)b[2]<<16) | ((uint32_t)b[3]<<24);
}
static inline uint16_t rd_u16le(FILE *f) {
    uint8_t b[2]; if (!fread_exact(b,2,f)) return 0;
    return (uint16_t)b[0] | ((uint16_t)b[1]<<8);
}

// static bool wav_read_header(FILE *f, wav_header_t *hdr, uint32_t *data_offset_out) {
//     if (!f || !hdr || !data_offset_out) return false;
//     memset(hdr, 0, sizeof(*hdr));

//     // RIFF + WAVE
//     if (!fread_exact(hdr->riff, 4, f)) return false;
//     if (memcmp(hdr->riff, "RIFF", 4) != 0) return false;

//     hdr->chunk_size = rd_u32le(f);

//     if (!fread_exact(hdr->wave, 4, f)) return false;
//     if (memcmp(hdr->wave, "WAVE", 4) != 0) return false;

//     bool have_fmt  = false;
//     bool have_data = false;
//     uint32_t data_offset = 0;

//     while (!have_fmt || !have_data) {
//         char id[4];
//         if (!fread_exact(id, 4, f)) return false;
//         uint32_t sz = rd_u32le(f);

//         if (memcmp(id, "fmt ", 4) == 0) {
//             memcpy(hdr->fmt, "fmt ", 4);
//             hdr->subchunk1_size = sz;
//             if (sz < 16) return false; // PCM base fmt

//             long start = ftell(f);
//             hdr->audio_format    = rd_u16le(f);
//             hdr->num_channels    = rd_u16le(f);
//             hdr->sample_rate     = rd_u32le(f);
//             hdr->byte_rate       = rd_u32le(f);
//             hdr->block_align     = rd_u16le(f);
//             hdr->bits_per_sample = rd_u16le(f);

//             // Skip any fmt extension
//             long consumed = ftell(f) - start;
//             if ((uint32_t)consumed < sz) fseek(f, sz - consumed, SEEK_CUR);

//             have_fmt = true;
//         } else if (memcmp(id, "data", 4) == 0) {
//             memcpy(hdr->data_header, "data", 4);
//             hdr->data_size = sz;
//             data_offset = (uint32_t)ftell(f);
//             fseek(f, sz, SEEK_CUR);
//             have_data = true;
//         } else {
//             // unknown chunk, skip
//             fseek(f, sz, SEEK_CUR);
//         }

//         // word-alignment pad for odd-sized chunks
//         if (sz & 1) fseek(f, 1, SEEK_CUR);
//     }

//     *data_offset_out = data_offset;
//     return true;
// }

static void slide_and_append(float *dst, int N, int H, const float *src_new, int count) {
    if (H > N) return;
    memmove(dst, dst + H, sizeof(float) * (N - H));
    memcpy(dst + (N - H), src_new, sizeof(float) * count);
    if (count < H) {
        memset(dst + (N - H) + count, 0, sizeof(float) * (H - count));
    }
}

// frequency band edges computed using np.geomspace(20,16000,8).round(0)
static const uint16_t band_edges_log[9] = {20, 46, 106, 245, 566, 1305, 3008, 6938, 16000};
void condense_mono_bins_log(void) {
    int sr = wav_hdr.sample_rate;
    for(size_t b = 0; b < N_BINS; b++) {
        int start = (int)(band_edges_log[b] * N_SAMPLES / (2.0f * sr));
        int end = (int)(band_edges_log[b+1] * N_SAMPLES / (2.0f * sr));
        if(end <= start) end = start+1;

        float sum = 0.0f;
        for(size_t i = start; i < end; i++) 
            sum += mono[i];
        bins8[b] = sum / (end - start); 
    }
}

static void process_fft_frame(void) {
    for (int i = 0; i < N_SAMPLES; i++) {
        float w = wind[i];
        y_cf[2*i + 0] = x1[i] * w;  // real = left
        y_cf[2*i + 1] = x2[i] * w;  // imag = right
    }

    dsps_fft2r_fc32(y_cf, N_SAMPLES);
    dsps_bit_rev_fc32(y_cf, N_SAMPLES);
    dsps_cplx2reC_fc32(y_cf, N_SAMPLES); // y1_cf & y2_cf now interleaved complex spectra

    // Convert to linear magnitudes and normalize 0..1 (per frame)
    float max_v = 1e-12f;
    for (int k = 0; k < N_FFT_BINS; k++) {
        float r1 = y1_cf[2*k+0], i1 = y1_cf[2*k+1];
        float r2 = y2_cf[2*k+0], i2 = y2_cf[2*k+1];
        float a1 = hypotf(r1, i1);
        float a2 = hypotf(r2, i2);
        mag1[k] = a1;
        mag2[k] = a2;
        if (a1 > max_v) max_v = a1;
        if (a2 > max_v) max_v = a2;
    }
    float inv = 1.0f / max_v;
    for (int k = 0; k < N_FFT_BINS; k++) {
        mono[k] = (mag1[k] + mag2[k])/2 * inv;
    }
    condense_mono_bins_log();
}



static i2s_chan_handle_t tx_chan;

static void init_i2s(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &tx_chan, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(wav_hdr.sample_rate),
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

static float tmpL[HOP] __attribute__((aligned(16)));  // If you prefer direct-tail fill, you can delete these.
static float tmpR[HOP] __attribute__((aligned(16)));

// typedef struct wav_args {
//     FILE *f;
//     wav_header_t *hdr;
// } wav_args;

// static void play_wav(void *args) {
//     wav_args *w_args = (wav_args*)args;
//     wav_header_t *hdr = w_args->hdr;
//     FILE *f = w_args->f;

//     const uint32_t bytes_per_frame = hdr->num_channels * 2; // 16-bit
//     const size_t chunk_bytes = HOP * bytes_per_frame;

// }

// static void fft_comm(void *args) {
//     wav_args* w_args = (wav_args*)args;
//     wav_header_t *hdr = w_args->hdr;
//     FILE *f = w_args->f;

//     const uint32_t bytes_per_frame = hdr->num_channels * 2; // 16-bit
//     const size_t chunk_bytes = HOP * bytes_per_frame;
// }

void print_raw_frames() {
    printf("Raw frequency bins: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
    bins8[0], bins8[1], bins8[2], bins8[3], bins8[4], bins8[5], bins8[6], bins8[7]);
    printf("\n");
}

void print_bar_frames() {
    for(size_t b = 0; b < N_BINS; b++) {
        int len = (int)(bins8[b]*10);
        for(int i = 0; i<12; i++) {
            printf("%c", (len >= 0) ? '=' : ' ');
            len--;
        }
        printf("\n");
    }
    printf("\n");
}

void print_text_frames() {
    for(size_t b = 0; b < N_BINS; b++) {
        printf("%s ", (bins8[b] > 0.05) ? "HIGH" : "LOW");
    }
    printf("\n");
}

static void play_wav_file(FILE *f) {
    ESP_LOGI(TAG, "Starting FFT transformation");
    
    float sr = (float)wav_hdr.sample_rate;
    int win = N_SAMPLES;
    int hop = HOP;
    int bins = N_SAMPLES / 2;
    float frame_ms = 1000.0f * win / sr;
    float hop_ms   = 1000.0f * hop / sr;
    float hz_per_bin = sr / (float)win;

    // Print the shape of the frame collected
    ESP_LOGI("FFTINFO",
            "Frame shape: channels=2, bins=%d per channel (total=%d); "
            "window=%d samples, hop=%d samples; "
            "frame=%.2f ms, hop=%.2f ms; %.2f Hz/bin",
            bins, 2*bins, win, hop, frame_ms, hop_ms, hz_per_bin);

    // vTaskDelay(pdMS_TO_TICKS(8000));

    if (dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE) != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed");
        return;
    }
    dsps_wind_hann_f32(wind, N_SAMPLES);

    uint8_t bytes_per_frame = wav_hdr.num_channels * 2;

    const int chunk_bytes = HOP * bytes_per_frame;

    uint8_t *pcm_buf = heap_caps_malloc(chunk_bytes, MALLOC_CAP_DMA);
    if (!pcm_buf) {
        ESP_LOGE(TAG, "Failed to alloc pcm_buf %u", (unsigned)chunk_bytes);
        return;
    }
    size_t bytes_read;
    size_t bytes_written;
    
    unsigned int frame_count = 0;

    while ((bytes_read = fread(pcm_buf, 1, chunk_bytes, f)) > 0) {
        frame_count++;
        // esp_err_t wr = i2s_channel_write(tx_chan, pcm_buf, bytes_read, &bytes_written, portMAX_DELAY);
        // if (wr == ESP_ERR_TIMEOUT) {
        //     ESP_LOGI(TAG, "ESP_ERR_TIMEOUT encountered, restarting");
        //     vTaskDelay(1);
        //     continue; // try again next loop
        // } else if (wr != ESP_OK) {
        //     ESP_LOGW(TAG, "i2s write err=%s wrote=%u of %u", esp_err_to_name(wr), (unsigned)bytes_written, (unsigned)bytes_read);
        //     // continue anyway; skip analysis this iteration
        //     vTaskDelay(1);
        //     continue;
        // }

        // Convert to float and slide into FFT frame
        const int samples_read = bytes_read / bytes_per_frame; // frames per channel
        // Safety: samples_read must be <= HOP because we set want<=chunk_bytes
        if (samples_read > HOP) {
            ESP_LOGW(TAG, "samples_read(%d) > HOP(%d) — clamping", samples_read, HOP);
        }

        for (int i = 0; i < HOP; i++) {
            if (i < samples_read) {
                const int16_t *p = (const int16_t *)pcm_buf + i * wav_hdr.num_channels;
                int16_t l = p[0];
                int16_t r = (wav_hdr.num_channels == 2) ? p[1] : p[0];
                tmpL[i] = (float)l / 32768.0f;
                tmpR[i] = (float)r / 32768.0f;
            } else {
                tmpL[i] = 0.0f;
                tmpR[i] = 0.0f;
            }
        }

        slide_and_append(x1, N_SAMPLES, HOP, tmpL, HOP);
        slide_and_append(x2, N_SAMPLES, HOP, tmpR, HOP);

        process_fft_frame();
        const char frame[] = "<FRAME> ";
        const char sep[] = "<SEP>";
        char text[sizeof(frame) + N_BINS * 5 + sizeof(sep)];

        int p = strlen(frame);     
        strcpy(text, frame);      

        for (size_t b = 0; b < N_BINS; b++) {
            if (bins8[b] > 0.05f) {
                strcpy(text + p, "HIGH ");
                p += 5;
            } else {
                strcpy(text + p, "LOW ");
                p += 4;
            }
        }

        strcpy(text + p, sep);
        text[p  + strlen(sep)] = '\0';
        printf("%s   %d\n", text, frame_count);
    }

    heap_caps_free(pcm_buf);
    gpio_set_level(led_pin, 0);

    ESP_LOGI(TAG, "All Done! %d frames", frame_count);
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(3000));

    ESP_LOGI(TAG, "Starting SD card bring-up test");
    
    
    init_sdcard();
    
    FILE *wav_f = fopen("/sdcard/001.wav", "rb");
    if (!wav_f) {
        ESP_LOGE(TAG, "File not found");
        return;
    }
    fread(&wav_hdr, sizeof(wav_hdr), 1, wav_f);
    ESP_LOGI("Main", "%d", (int)wav_hdr.sample_rate);
    
    init_i2s();

    gpio_reset_pin(led_pin);
    gpio_set_direction(led_pin, GPIO_MODE_OUTPUT);
    gpio_set_level(led_pin, 1);

    // xTaskCreatePinnedToCore(
    //     play_wav,        // Task function
    //     "wav file player",           // Task name
    //     4096,               // Stack size (in bytes)
    //     NULL,               // Task parameter
    //     1,                  // Task priority
    //     NULL,               // Task handle
    //     0                   // Core ID (0 for CPU0)
    // );

    // xTaskCreatePinnedToCore(
    //     fft_comm,              // Task function
    //     "fft",           // Task name
    //     4096,               // Stack size (in bytes)
    //     NULL,               // Task parameter
    //     1,                  // Task priority
    //     NULL,               // Task handle
    //     1                   // Core ID (1 for CPU1)
    // );
    
    ESP_LOGI(TAG, "%d", configNUM_CORES);


    play_wav_file(wav_f);

    fclose(wav_f);
}
