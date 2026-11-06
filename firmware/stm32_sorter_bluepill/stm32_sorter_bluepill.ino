// STM32 colour sorter - Blue Pill (STM32F103C8T6), STM32duino core.
//
// Changes from the original firmware:
//  * EEPROM removed (the Blue Pill has no EEPROM). The last classification is kept in RAM only,
//    so STATUS reports LAST_SAVED:NONE after every power-up.
//  * "BIN:<name>" is printed whenever the colour under the sensor changes, also while idle,
//    so the Python GUI can show which bin the sensor is over. Python ignores these lines
//    when it waits for a reply.

#include <Arduino.h>
#include <math.h>

// ==========================================
// 1. MOTOR PINS - L298N
// ==========================================

// Platform motor
#define MOTOR1_IN1 A7
#define MOTOR1_IN2 A6

// Gate motor
#define MOTOR2_IN1 A0
#define MOTOR2_IN2 A1

// ==========================================
// 2. TCS3200 PINS
// ==========================================

#define S0 B0
#define S1 B1
#define S2 B5
#define S3 B6
#define SENSOR_OUT B7

// ==========================================
// 3. SETTINGS
// ==========================================

const unsigned long ROTATION_TIMEOUT = 20000;

const unsigned long GATE_MOVE_TIME = 800;
const unsigned long GATE_WAIT_TIME = 3000;

// While idle, the colour under the sensor is checked this often (ms)
const unsigned long BIN_CHECK_INTERVAL = 300;

const int STABLE_READINGS = 3;

// ==========================================
// 4. MATERIAL IDs
// ==========================================

enum Material {
  UNKNOWN = 0,
  METAL = 1,
  PLASTIC = 2,
  PAPER = 3,
  ORGANIC = 4
};

// ==========================================
// 5. COLOR DATA
// ==========================================

struct ColorData {
  float R;
  float G;
  float B;
};

// ==========================================
// 6. COLOR RANGES
// ==========================================

// RED = METAL
const float RED_R_MIN = 44;
const float RED_R_MAX = 64;

const float RED_G_MIN = 16;
const float RED_G_MAX = 26;

const float RED_B_MIN = 18;
const float RED_B_MAX = 30;


// BLUE = PLASTIC
// Provisional range based on the more stable readings
const float BLUE_R_MIN = 30;
const float BLUE_R_MAX = 36;

const float BLUE_G_MIN = 28;
const float BLUE_G_MAX = 33;

const float BLUE_B_MIN = 33;
const float BLUE_B_MAX = 38;


// PURPLE = PAPER
const float PURPLE_R_MIN = 35;
const float PURPLE_R_MAX = 41;

const float PURPLE_G_MIN = 22;
const float PURPLE_G_MAX = 27;

const float PURPLE_B_MIN = 35;
const float PURPLE_B_MAX = 40;


// YELLOW = ORGANIC
// Add the measured yellow ranges when available.
// Yellow classification is currently disabled.

// ==========================================
// 7. SYSTEM STATES
// ==========================================

enum SystemState {
  IDLE,
  ROTATING,
  GATE_OPENING,
  GATE_WAITING,
  GATE_CLOSING
};

SystemState currentState = IDLE;

// ==========================================
// 8. VARIABLES
// ==========================================

Material requestedMaterial = UNKNOWN;
Material detectedMaterial = UNKNOWN;
Material lastSavedMaterial = UNKNOWN;

Material candidateMaterial = UNKNOWN;

int stableCount = 0;

bool markerArmed = true;

unsigned long rotationStartTime = 0;
unsigned long stateStartTime = 0;
unsigned long lastBinCheck = 0;

String inputLine = "";

// ==========================================
// 9. MOTOR FUNCTIONS
// ==========================================

void stopPlatform() {

  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
}

void rotatePlatform() {

  digitalWrite(MOTOR1_IN1, HIGH);
  digitalWrite(MOTOR1_IN2, LOW);
}

void stopGate() {

  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);
}

void openGate() {

  digitalWrite(MOTOR2_IN1, HIGH);
  digitalWrite(MOTOR2_IN2, LOW);
}

void closeGate() {

  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, HIGH);
}

// ==========================================
// 10. READ SENSOR
// ==========================================

unsigned long readPulse(bool s2, bool s3) {

  digitalWrite(S2, s2);
  digitalWrite(S3, s3);

  delay(5);

  return pulseIn(SENSOR_OUT, LOW, 30000);
}


float readFrequency(bool s2, bool s3) {

  unsigned long pulse = readPulse(s2, s3);

  if (pulse == 0) {
    return 0;
  }

  return 1000000.0 / pulse;
}


// ==========================================
// 11. READ RGB PERCENTAGES
// ==========================================

ColorData readColor() {

  ColorData color;

  float red = 0;
  float green = 0;
  float blue = 0;

  // Take 5 readings
  for (int i = 0; i < 5; i++) {

    red += readFrequency(LOW, LOW);

    blue += readFrequency(LOW, HIGH);

    green += readFrequency(HIGH, HIGH);
  }

  red /= 5.0;
  green /= 5.0;
  blue /= 5.0;

  float total = red + green + blue;

  if (total <= 0) {

    color.R = 0;
    color.G = 0;
    color.B = 0;

    return color;
  }

  // Convert to percentages

  color.R = (red / total) * 100.0;

  color.G = (green / total) * 100.0;

  color.B = (blue / total) * 100.0;

  return color;
}

// ==========================================
// 12. CHECK COLOR RANGE
// ==========================================

bool insideRange(
  ColorData c,
  float rMin, float rMax,
  float gMin, float gMax,
  float bMin, float bMax
) {

  return (
    c.R >= rMin && c.R <= rMax &&
    c.G >= gMin && c.G <= gMax &&
    c.B >= bMin && c.B <= bMax
  );
}

// ==========================================
// 13. CLASSIFY COLOR
// ==========================================

Material classifyColor(ColorData c) {

  if (c.R == 0 && c.G == 0 && c.B == 0) {
    return UNKNOWN;
  }

  // RED = METAL
  if (insideRange(
    c,
    RED_R_MIN, RED_R_MAX,
    RED_G_MIN, RED_G_MAX,
    RED_B_MIN, RED_B_MAX
  )) {

    return METAL;
  }

  // BLUE = PLASTIC
  if (insideRange(
    c,
    BLUE_R_MIN, BLUE_R_MAX,
    BLUE_G_MIN, BLUE_G_MAX,
    BLUE_B_MIN, BLUE_B_MAX
  )) {

    return PLASTIC;
  }

  // PURPLE = PAPER
  if (insideRange(
    c,
    PURPLE_R_MIN, PURPLE_R_MAX,
    PURPLE_G_MIN, PURPLE_G_MAX,
    PURPLE_B_MIN, PURPLE_B_MAX
  )) {

    return PAPER;
  }

  // YELLOW = ORGANIC
  // Add its measured RGB range here later.

  return UNKNOWN;
}

// ==========================================
// 14. MATERIAL NAME
// ==========================================

const char* getMaterialName(Material material) {

  switch (material) {

    case METAL:
      return "METAL";

    case PLASTIC:
      return "PLASTIC";

    case PAPER:
      return "PAPER";

    case ORGANIC:
      return "ORGANIC";

    default:
      return "UNKNOWN";
  }
}

// ==========================================
// 15. SAVE LAST CLASSIFICATION (RAM ONLY)
// ==========================================

void saveLastClassification(Material material) {

  if (material == UNKNOWN) {
    return;
  }

  lastSavedMaterial = material;
}


// ==========================================
// 16. LOAD LAST CLASSIFICATION (RAM ONLY)
// ==========================================

void loadLastClassification() {

  // Nothing is kept across power cycles on the Blue Pill
  lastSavedMaterial = UNKNOWN;
}

// ==========================================
// 16b. BIN REPORT
// ==========================================

// The bin (colour marker) the sensor is over right now. Printed as
// "BIN:METAL", "BIN:PLASTIC", "BIN:PAPER", "BIN:ORGANIC" or "BIN:NONE",
// and only when it changes.
int lastBinReported = -1;

void reportBin(Material material) {

  if ((int)material == lastBinReported) {
    return;
  }

  lastBinReported = (int)material;

  Serial.print("BIN:");

  if (material == UNKNOWN) {
    Serial.println("NONE");
  }
  else {
    Serial.println(getMaterialName(material));
  }
}

// ==========================================
// 17. PROCESS SENSOR READING
// ==========================================

void processSensor() {

  ColorData color = readColor();

  Material currentColor = classifyColor(color);

  // Tell Python which bin the sensor is over now
  reportBin(currentColor);

  // No valid color detected
  if (currentColor == UNKNOWN) {

    stableCount = 0;

    candidateMaterial = UNKNOWN;

    markerArmed = true;

    return;
  }

  // Check if the same color is detected repeatedly
  if (currentColor == candidateMaterial) {

    stableCount++;

  }

  else {

    candidateMaterial = currentColor;

    stableCount = 1;
  }

  // Require stable readings
  if (stableCount < STABLE_READINGS) {
    return;
  }

  // Do not process the same marker repeatedly
  if (!markerArmed) {
    return;
  }

  markerArmed = false;

  detectedMaterial = currentColor;

  // Remember the last classification (RAM only)
  saveLastClassification(detectedMaterial);

  // Send material name to Python
  Serial.println(getMaterialName(detectedMaterial));

  // Check if this is the requested destination
  if (detectedMaterial == requestedMaterial) {

    stopPlatform();

    Serial.println("POSITION CONFIRMED");

    openGate();

    stateStartTime = millis();

    currentState = GATE_OPENING;
  }
}

// ==========================================
// 18. START SORTING
// ==========================================

void startSorting(Material material) {

  if (currentState != IDLE) {

    Serial.println("BUSY");

    return;
  }

  requestedMaterial = material;

  candidateMaterial = UNKNOWN;

  stableCount = 0;

  markerArmed = true;

  rotationStartTime = millis();

  Serial.print("REQUESTED:");

  Serial.println(getMaterialName(material));

  rotatePlatform();

  currentState = ROTATING;
}

// ==========================================
// 19. SERIAL COMMANDS
// ==========================================

void processCommand(String command) {

  command.trim();

  command.toUpperCase();

  if (command == "1") {

    startSorting(METAL);
  }

  else if (command == "2") {

    startSorting(PLASTIC);
  }

  else if (command == "3") {

    startSorting(PAPER);
  }

  else if (command == "4") {

    startSorting(ORGANIC);
  }

  else if (command == "STATUS") {

    Serial.print("LAST_SAVED:");

    Serial.println(
      getMaterialName(lastSavedMaterial)
    );
  }

  else if (command == "STOP") {

    stopPlatform();

    stopGate();

    currentState = IDLE;

    requestedMaterial = UNKNOWN;

    Serial.println("STOPPED");
  }

  else {

    Serial.println("INVALID COMMAND");
  }
}

// ==========================================
// 20. READ SERIAL
// ==========================================

void readSerial() {

  while (Serial.available()) {

    char incoming = Serial.read();

    if (incoming == '\n') {

      processCommand(inputLine);

      inputLine = "";
    }

    else if (incoming != '\r') {

      if (inputLine.length() < 30) {

        inputLine += incoming;
      }
    }
  }
}

// ==========================================
// 21. UPDATE SYSTEM
// ==========================================

void updateSystem() {

  unsigned long now = millis();

  // WHILE IDLE: keep checking which bin the sensor is over
  if (currentState == IDLE && now - lastBinCheck >= BIN_CHECK_INTERVAL) {

    lastBinCheck = now;

    reportBin(classifyColor(readColor()));
  }

  // PLATFORM ROTATION
  if (currentState == ROTATING) {

    if (now - rotationStartTime >= ROTATION_TIMEOUT) {

      stopPlatform();

      currentState = IDLE;

      requestedMaterial = UNKNOWN;

      Serial.println("POSITION TIMEOUT");

      return;
    }

    processSensor();
  }

  // GATE OPENING
  else if (currentState == GATE_OPENING) {

    if (now - stateStartTime >= GATE_MOVE_TIME) {

      stopGate();

      stateStartTime = millis();

      currentState = GATE_WAITING;

      Serial.println("GATE OPEN");
    }
  }

  // WAIT 3 SECONDS
  else if (currentState == GATE_WAITING) {

    if (now - stateStartTime >= GATE_WAIT_TIME) {

      closeGate();

      stateStartTime = millis();

      currentState = GATE_CLOSING;
    }
  }

  // GATE CLOSING
  else if (currentState == GATE_CLOSING) {

    if (now - stateStartTime >= GATE_MOVE_TIME) {

      stopGate();

      currentState = IDLE;

      requestedMaterial = UNKNOWN;

      Serial.println("GATE CLOSED");

      Serial.println("SORTING COMPLETE");
    }
  }
}

// ==========================================
// 22. SETUP
// ==========================================

void setup() {

  Serial.begin(115200);

  // Motor pins
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);

  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);

  // Sensor pins
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);

  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);

  pinMode(SENSOR_OUT, INPUT);

  // TCS3200 frequency scaling: 20%
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);

  stopPlatform();

  stopGate();

  loadLastClassification();  // RAM only: starts as UNKNOWN

  Serial.println("SYSTEM READY");

  Serial.print("LAST SAVED CLASSIFICATION:");

  Serial.println(
    getMaterialName(lastSavedMaterial)
  );

  Serial.println("SEND 1, 2, 3 OR 4 TO START");
}

// ==========================================
// 23. MAIN LOOP
// ==========================================

void loop() {

  readSerial();

  updateSystem();
}