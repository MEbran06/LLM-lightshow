#include <stdio.h>
#include <inttypes.h>
#include "esp_spiffs.h"
#include "sdkconfig.h"
#include "esp_err.h"
#include "esp_log.h"
#include <time.h>
#include "llm.h"
#include <driver/i2c.h>
#include <string.h>
#include "llama.h"
#include "driver/gpio.h"
#include <dirent.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SER_PIN     GPIO_NUM_13   // Data
#define SRCLK_PIN   GPIO_NUM_12   // Shift clock
#define RCLK_PIN    GPIO_NUM_11   // Latch clock

static const char *TAG = "MAIN";
static const gpio_num_t led_pin = GPIO_NUM_4;

/**
 * @brief intializes SPIFFS storage
 * 
 */
int init_storage(void)
{

    ESP_LOGI(TAG, "Initializing SPIFFS");

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/data",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false};

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGI(TAG, "Failed to mount or format filesystem");
        }
        else if (ret == ESP_ERR_NOT_FOUND)
        {
            ESP_LOGI(TAG, "Failed to find SPIFFS partition");
        }
        else
        {
            ESP_LOGI(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }
        return 1;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret != ESP_OK)
    {
        ESP_LOGI(TAG, "Failed to get SPIFFS partition information (%s)", esp_err_to_name(ret));
        return 1;
    }
    else
    {
        ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
        printf("SPIFFS mounted at: %s\n", conf.base_path);
        return 0;
    }

    return 0;
}

/**
 * @brief Callbacks once generation is done
 * 
 * @param tk_s The number of tokens per second generated
 */
void generate_complete_cb(float tk_s)
{
    char buffer[50];
    sprintf(buffer, "%.2f tok/s", tk_s);
}

static void shift_out(uint8_t data)
{
    // set Latch to low (lets us write the bits)
    gpio_set_level(RCLK_PIN, 0);

    for (int i = 7; i >= 0; i--) {
        // pass a bit (MSB first)
        int bit = (data >> i) & 1;
        // pass set the LED to whatever that bit is
        gpio_set_level(SER_PIN, bit);

        // move the bits inside internal shift register
        gpio_set_level(SRCLK_PIN, 0);
        gpio_set_level(SRCLK_PIN, 1);
    }

    // update the output pins (actually turn them on)
    gpio_set_level(RCLK_PIN, 1);
    gpio_set_level(RCLK_PIN, 0);
}


void app_main(void)
{
    //init_display();
    //write_display("Loading Model");
    gpio_reset_pin(led_pin);
    gpio_reset_pin(SER_PIN);
    gpio_reset_pin(SRCLK_PIN);
    gpio_reset_pin(RCLK_PIN);
    gpio_set_direction(SER_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(SRCLK_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(RCLK_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(led_pin, GPIO_MODE_OUTPUT);
    if (init_storage() == 1)
    {
        printf("INITIALIZING SPIFF FAILED!!!\n");
        gpio_set_level(led_pin, 1);
    }

    
    // default parameters
    char *checkpoint_path = "/data/model.bin"; // e.g. out/model.bin
    char *tokenizer_path = "/data/tokenizer.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;                 // number of steps to run for
    char *prompt = "<FRAME> HIGH LOW LOW HIGH LOW LOW HIGH LOW <SEP>";             // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    Transformer transformer;
    ESP_LOGI(TAG, "LLM Path is %s", checkpoint_path);
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    char** gen_seq = generate(&transformer, &tokenizer, &sampler, prompt, steps, &generate_complete_cb);

    int bit_count = 7;
    uint8_t total = 0;
    for (int i=0; i<24; i++)
    {
        if (gen_seq[i][0] == '0')
        {
           bit_count--;
        }   
        else if (atoi(gen_seq[i]) == 1)
        {
            total += 1 << bit_count;
            bit_count--;
        }
    }
    printf("\nTOTAL: %d\n", total);
    shift_out(total);
}
