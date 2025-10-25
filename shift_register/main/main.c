#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SER_PIN     GPIO_NUM_13   // Data
#define SRCLK_PIN   GPIO_NUM_12   // Shift clock
#define RCLK_PIN    GPIO_NUM_11   // Latch clock

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
    // set each pin as output
    gpio_set_direction(SER_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(SRCLK_PIN, GPIO_MODE_OUTPUT);
    gpio_set_direction(RCLK_PIN, GPIO_MODE_OUTPUT);

    while (true) {
        // shift out a byte
        shift_out(0b10101010);
        // delay
        vTaskDelay(pdMS_TO_TICKS(1000));
        // shift out a byte
        shift_out(0b01010101);
        // delay
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
