#include <Wire.h>
#include "Adafruit_VL53L1X.h"

#define TCAADDR 0x70

Adafruit_VL53L1X vl53 = Adafruit_VL53L1X();

// Array of active channels
uint8_t channels[] = {0, 1, 2, 3, 4, 5, 6, 7};
const uint8_t numChannels = sizeof(channels) / sizeof(channels[0]);

void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  delay(1000);

  Wire.begin();

  Serial.println("Initializing sensors on all channels");

  for (uint8_t c = 0; c < numChannels; c++) {
    uint8_t channel = channels[c];
    tcaselect(channel);
    delay(400);
    Serial.print("Testing channel: ");
    Serial.println(channels[c]);
    if (!vl53.begin(0x29, &Wire)) {
      Serial.print("Sensor on channel "); Serial.print(channel); Serial.println(" not found");
    } else {
      Serial.print("Sensor on channel "); Serial.print(channel); Serial.println(" initialized");
      vl53.startRanging();
    }
    delay(100);
  }

  Serial.println("Setup complete.");
}

void loop() {
  for (uint8_t c = 0; c < numChannels; c++) {
    uint8_t channel = channels[c];
    tcaselect(channel);

    if (vl53.dataReady()) {
      uint16_t distance = vl53.distance();
      Serial.print("Sensor "); Serial.print(channel); Serial.print(": ");
      Serial.print(distance); Serial.println(" mm");
      vl53.clearInterrupt();
    } else {
      Serial.print("Sensor "); Serial.print(channel); Serial.println(": Data not ready");
    }

    delay(50);
  }

  Serial.println("---");
  delay(500);
}
