idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp-dsp esp_driver_i2s esp_driver_sdspi sdmmc fatfs
)