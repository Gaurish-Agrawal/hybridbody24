#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "Adafruit_VL53L1X.h"

// === Wi-Fi Credentials ===
const char* ssid = "wurc2.4";
const char* password = "robotics@washu";

// === IP address of the motor control ESP32 ===
const char* motorControllerIP = "192.168.1.44";  // <-- Replace with actual IP

// === I2C Multiplexer Address ===
#define TCAADDR 0x70

// === Sensor Setup ===
Adafruit_VL53L1X vl53 = Adafruit_VL53L1X();
uint8_t channels[] = {0, 1, 6, 7};
const char* directions[] = {"N", "S", "E", "W"};
const uint8_t numChannels = sizeof(channels) / sizeof(channels[0]);

// === Distance → Intensity Mapping ===
const int maxDistance = 1000; // mm
const int minDistance = 10;  // mm
const int minIntensity = 1;
const int maxIntensity = 127;

// === I2C Channel Select ===
void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());

  // Initialize each sensor
  for (uint8_t c = 0; c < numChannels; c++) {
    uint8_t channel = channels[c];
    tcaselect(channel);
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
    const char* direction = directions[c];
    tcaselect(channel);

    if (vl53.dataReady()) {
      uint16_t distance = vl53.distance();
      vl53.clearInterrupt();

      int intensity = map(constrain(distance, minDistance, maxDistance),
                        minDistance, maxDistance,
                        maxIntensity, minIntensity);

      float normalized = (float)(intensity - minIntensity) / (maxIntensity - minIntensity);
      if (distance < 20000) {    
        uint8_t effectID = intensityToEffectID(normalized);
        
        Serial.print("Sensor "); Serial.print(channel); Serial.print(" (");
        Serial.print(direction); Serial.print("): ");
        Serial.print(distance); Serial.print(" mm -> Intensity: ");
        Serial.print(intensity); Serial.print(" -> EffectID: ");
        Serial.println(effectID);

        sendToMotorController(direction, effectID);  // <-- Send effect ID instead of raw intensity  
      }
      sendToVisualizer(direction, distance);
    } else {
      Serial.print("Sensor "); Serial.print(channel); Serial.println(": Data not ready");
    }

    delay(50);
  }

  Serial.println("---");
  delay(500);
}

uint8_t intensityToEffectID(float normalizedIntensity) {
  if (normalizedIntensity < 0.0 || normalizedIntensity > 1.0) {
    return 1;  // default safe value
  } else if (normalizedIntensity <= 0.25) {
    return 50;
  } else if (normalizedIntensity <= 0.5) {
    return 49;
  } else if (normalizedIntensity <= 0.75) {
    return 48;
  } else {
    return 47;
  }
}


void sendToMotorController(const char* direction, int intensity) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  String url = "http://" + String(motorControllerIP) + "/" + direction + "?value=" + String(intensity);
  http.begin(url);
  int httpCode = http.GET();  // Send GET request

  if (httpCode > 0) {
    Serial.print("Sent to "); Serial.print(direction);
    Serial.print(" | Response: "); Serial.println(httpCode);
  } else {
    Serial.print("Failed to send to "); Serial.print(direction);
    Serial.print(" | Error: "); Serial.println(http.errorToString(httpCode).c_str());
  }

  http.end();
}

void sendToVisualizer(const char* direction, uint16_t distance) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  String url = "http://192.168.1.17:5000/update";  // Replace with computer's IP and port

  // Construct simple JSON
  String payload = "{\"dir\":\"" + String(direction) + "\",\"dist\":" + String(distance) + "}";

  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    Serial.print("Sent to visualizer: "); Serial.println(payload);
  } else {
    Serial.print("Failed to send to visualizer. Error: ");
    Serial.println(http.errorToString(httpCode).c_str());
  }

  http.end();
}

