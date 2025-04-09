#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <VL53L1X.h>

// === Wi-Fi Credentials ===
const char* ssid = "wurc2.4";
const char* password = "robotics@washu";

#define TCA_ADDR 0x70 // TCA9548A address
#define NUM_SENSORS 4 // Number of sensors

VL53L1X sensors[NUM_SENSORS];

// Channels to use: 0, 1, 6, 7
uint8_t muxChannels[NUM_SENSORS] = {0, 1, 6, 7};
const char* directions[] = {"N", "E", "W", "S"};

void tcaSelect(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());

  for (int i = 0; i < NUM_SENSORS; i++) {
    tcaSelect(muxChannels[i]);
    delay(10);

    sensors[i].setTimeout(500);
    if (!sensors[i].init()) {
      Serial.print("Failed to detect sensor at multiplexer channel ");
      Serial.println(muxChannels[i]);
      continue;
    }

    sensors[i].setDistanceMode(VL53L1X::Unknown);
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
    const char* direction = directions[i];

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

    sendToVisualizer(direction, dist);
  }

  Serial.println("-----");
  delay(300);
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
}
