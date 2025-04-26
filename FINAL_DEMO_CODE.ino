#include <Wire.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <ArduinoHttpClient.h>
#include "Adafruit_VL53L1X.h"

#define TCAADDR 0x70
#define THRESHOLD_MM 50

// Access Point credentials
char ssid[] = "PortentaAccess";
char pass[] = "1234abcd";

// Target client (connected to AP)
const char* targetIP = "192.168.3.2";  // IP of laptop connected to AP
int targetPort = 5000;

WiFiClient wifiClient;
HttpClient httpClient = HttpClient(wifiClient, targetIP, targetPort);

// Sensors and motors setup
Adafruit_VL53L1X vl53 = Adafruit_VL53L1X();
uint8_t sensorChannels[] = {0, 1, 2, 3, 4, 5, 6, 7};  // N, NE, E, SE, S, SW, W, NW
uint8_t motorPins[]     = {0, 1, 2, 3, 4, 5, 6, 7};    // NE, N, E, SE, S, SW, W, NW
const uint8_t numSensors = sizeof(sensorChannels) / sizeof(sensorChannels[0]);

int status = WL_IDLE_STATUS;

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

  // Setup motor pins
  for (uint8_t i = 0; i < numSensors; i++) {
    pinMode(motorPins[i], OUTPUT);
    digitalWrite(motorPins[i], LOW);
  }

  // Initialize sensors
  for (uint8_t i = 0; i < numSensors; i++) {
    tcaselect(sensorChannels[i]);
    delay(400);
    if (!vl53.begin(0x29, &Wire)) {
      Serial.print("Sensor "); Serial.print(sensorChannels[i]); Serial.println(" not found");
    } else {
      vl53.startRanging();
      Serial.print("Sensor "); Serial.print(sensorChannels[i]); Serial.println(" initialized");
    }
  }

  // Start Access Point
  Serial.println("==================================================");
  Serial.print("Attempting to start AP: "); Serial.println(ssid);
  status = WiFi.beginAP(ssid, pass);
  delay(5000);

  if (status != WL_AP_LISTENING) {
    Serial.println("❌ Failed to start Access Point!");
  } else {
    Serial.println("✅ Access Point started successfully.");
  }

  Serial.print("Portenta AP IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println("==================================================");
}

void loop() {
  for (uint8_t i = 0; i < numSensors; i++) {
    uint8_t sensorChan = sensorChannels[i];
    uint8_t motorPin = motorPins[i];

    tcaselect(sensorChan);

    if (vl53.dataReady()) {
      uint16_t distance = vl53.distance();
      vl53.clearInterrupt();

      Serial.print("Sensor "); Serial.print(sensorChan); Serial.print(" (");
      Serial.print(motorPin); Serial.print(") distance: ");
      Serial.print(distance); Serial.println(" mm");

      // Control motor
      if (motorPin == 7) {
        // Pin 7 (NW) no PWM
        if (distance < 40) {
          digitalWrite(motorPin, HIGH);
        } else {
          digitalWrite(motorPin, LOW);
        }
      } else {
        if (distance < 10) {
          analogWrite(motorPin, 255); // Full power
        } else if (distance < 40) {
          analogWrite(motorPin, 128); // Half power
        } else {
          analogWrite(motorPin, 0);   // Off
        }
      }

      // Send HTTP GET to client
      if (WiFi.status() == WL_AP_CONNECTED) {
        String path = "/sensorUpdate?channel=" + getDirectionName(i) + "&distance=" + String(distance);

        Serial.print("📡 Sending to client: ");
        Serial.println(path);

        httpClient.beginRequest();
        httpClient.get(path);
        httpClient.endRequest();

        int statusCode = httpClient.responseStatusCode();
        String response = httpClient.responseBody();

        Serial.print("HTTP status: ");
        Serial.println(statusCode);
        Serial.println("Response:");
        Serial.println(response);

        wifiClient.stop();  // 🔥 Critical: reset TCP connection after each request
      } else {
        Serial.println("⚠️ No client connected yet.");
      }

    } else {
      Serial.print("Sensor "); Serial.print(sensorChan); Serial.println(": no data");
      digitalWrite(motorPin, LOW);
    }

    delay(100);
  }

  delay(300); // Extra time between full rounds
}

String getDirectionName(uint8_t sensorIndex) {
  switch (sensorIndex) {
    case 0: return "NW";
    case 1: return "N";
    case 2: return "W";
    case 3: return "S";
    case 4: return "SE";
    case 5: return "NE";
    case 6: return "E";
    case 7: return "SW";
    default: return "UNK";
  }
}

