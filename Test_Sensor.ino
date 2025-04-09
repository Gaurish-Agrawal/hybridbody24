#include <Wire.h>
#include <VL53L1X.h>

#define TCA_ADDR 0x70 // TCA9548A address
#define NUM_SENSORS 4 // Number of sensors

VL53L1X sensors[NUM_SENSORS];

// Channels to use: 0, 1, 6, 7
uint8_t muxChannels[NUM_SENSORS] = {0, 1, 6, 7};

void tcaSelect(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  for (int i = 0; i < NUM_SENSORS; i++) {
    tcaSelect(muxChannels[i]);
    delay(10);

    sensors[i].setTimeout(500);
    if (!sensors[i].init()) {
      Serial.print("Failed to detect sensor at multiplexer channel ");
      Serial.println(muxChannels[i]);
      continue;
    }

    sensors[i].setDistanceMode(VL53L1X::Long);
    sensors[i].setMeasurementTimingBudget(50000);
    sensors[i].startContinuous(50);

    Serial.print("Sensor initialized at multiplexer channel ");
    Serial.println(muxChannels[i]);
  }
}

void loop() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    tcaSelect(muxChannels[i]);
    delay(5);

    int dist = sensors[i].read();
    if (sensors[i].timeoutOccurred()) {
      Serial.print("Sensor at channel ");
      Serial.print(muxChannels[i]);
      Serial.println(" timeout!");
    } else {
      Serial.print("Sensor at channel ");
      Serial.print(muxChannels[i]);
      Serial.print(": ");
      Serial.print(dist);
      Serial.println(" mm");
    }
  }

  Serial.println("-----");
  delay(300);
}
