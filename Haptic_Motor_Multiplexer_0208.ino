#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define TCA9548A_ADDR 0x70  // Default address for TCA9548A
Adafruit_DRV2605 drv; // Create a DRV2605L instance

// Function to select TCA9548A I2C channel
void selectChannel(uint8_t channel) {
    if (channel > 7) return;  // Ensure valid channel range
    Wire.beginTransmission(TCA9548A_ADDR);
    Wire.write(1 << channel);  // Enable specific channel (bit shift)
    Wire.endTransmission();
}

void setup() {
    Serial.begin(115200);
    Wire.begin();

    Serial.println("Initializing DRV2605L on all 8 channels...");

    for (uint8_t i = 0; i < 8; i++) { // Loop through all 8 channels
        selectChannel(i);
        if (!drv.begin()) {
            Serial.print("DRV2605L not found on channel ");
            Serial.println(i);
        } else {
            Serial.print("DRV2605L initialized on channel ");
            Serial.println(i);
        }
        drv.selectLibrary(1); // Load haptic effect library
        drv.setMode(DRV2605_MODE_INTTRIG); // Set to internal trigger mode
    }
    Serial.println("Initialization Complete.");
}

void loop() {
    Serial.println("Testing all 8 haptic motors...");

    for (uint8_t i = 0; i < 8; i++) {
        Serial.print("Activating Motor on Channel ");
        Serial.println(i);

        selectChannel(i);
        drv.setWaveform(0, i + 10);  // Assign a different effect for each motor
        drv.setWaveform(1, 0);   // End sequence
        drv.go();

        delay(1000); // Wait before switching to next motor
    }
}
