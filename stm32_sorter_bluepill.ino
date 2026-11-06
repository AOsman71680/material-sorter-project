
/*
  SMART RECYCLING SORTER
  STM32F103C8T6 (BLUE PILL)
  TCS3200 + L298N + 2 DC MOTORS

  COLOR MAPPING:
  METAL    = RED
  PLASTIC  = BLUE
  PAPER    = PURPLE
  ORGANIC  = YELLOW

  MOTOR 1: Rotating platform
  MOTOR 2: Gate
*/

// ==================================================
// 1. PIN DEFINITIONS - STM32 BLUE PILL
// ==================================================

// TCS3200 COLOR SENSOR
#define S0          PB_0
#define S1          PB_1
#define S2          PB_4
#define S3          PB_3
#define SENSOR_OUT  PA_15

// MOTOR 1: ROTATING PLATFORM
#define ENA         PB_7
#define IN1         PA_7
#define IN2         PA_6

// MOTOR 2: GATE
#define ENB         PA_11
#define IN3         PA_0
#define IN4         PA_1

// ==================================================
// 2. SETTINGS
// ==================================================

#define ROTATION_SPEED  75
#define GATE_SPEED      255

#define GATE_TIME       5000UL
#define MAX_SEARCH_TIME 30000UL

#define SENSOR_TIMEOUT  100000UL

// ==================================================
// 3. GLOBAL VARIABLES
// ==================================================

String targetColor = "";
String detectedColor = "";


// ==================================================
// 4. READ COLOR SENSOR CHANNEL
// ==================================================

unsigned long readChannel(bool s2, bool s3) {

  digitalWrite(S2, s2);
  digitalWrite(S3, s3);

  delay(40);

  unsigned long pulse = pulseIn(
    SENSOR_OUT,
    LOW,
    SENSOR_TIMEOUT
  );

  // Zero means no valid pulse was received.
  return pulse;
}


// ==================================================
// 5. COLOR CLASSIFICATION
// ==================================================

String detectColor(float r, float g, float b) {

  if (r + g + b < 1) {
    return "NO COLOR";
  }

  // YELLOW - ORGANIC
  if (b < 24 && g > 32) {
    return "YELLOW";
  }

  // RED - METAL
  if (r > 42 && b < 30) {
    return "RED";
  }

  // PURPLE - PAPER
  if (g < 26 && r > 28 && b > 33) {
    return "PURPLE";
  }

  // BLUE - PLASTIC
  if (b >= r && b >= g) {
    return "BLUE";
  }

  return "UNKNOWN";
}


// ==================================================
// 6. READ BOX COLOR
// ==================================================

String readBoxColor() {

  unsigned long pr = readChannel(LOW, LOW);
  unsigned long pg = readChannel(HIGH, HIGH);
  unsigned long pb = readChannel(LOW, HIGH);

  // Check whether all channels returned valid readings.
  if (pr == 0 || pg == 0 || pb == 0) {
    Serial.println("Sensor reading error!");
    return "NO COLOR";
  }

  // Convert pulse periods into relative intensities.
  float ir = 1000000.0 / pr;
  float ig = 1000000.0 / pg;
  float ib = 1000000.0 / pb;

  float total = ir + ig + ib;

  if (total <= 0) {
    return "NO COLOR";
  }

  // Convert intensities into percentages.
  float r = 100.0 * ir / total;
  float g = 100.0 * ig / total;
  float b = 100.0 * ib / total;

  Serial.print("R: ");
  Serial.print(r, 1);

  Serial.print(" | G: ");
  Serial.print(g, 1);

  Serial.print(" | B: ");
  Serial.print(b, 1);

  String color = detectColor(r, g, b);

  Serial.print(" | Detected box: ");
  Serial.println(color);

  return color;
}


// ==================================================
// 7. MOTOR 1 - ROTATING PLATFORM
// ==================================================

void startRotation() {

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  analogWrite(ENA, ROTATION_SPEED);
}


void stopRotation() {

  analogWrite(ENA, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}


// ==================================================
// 8. MOTOR 2 - GATE
// ==================================================

void openGate() {

  Serial.println("Opening gate...");

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(ENB, GATE_SPEED);

  delay(GATE_TIME);

  // Stop the gate motor after 5 seconds.
  analogWrite(ENB, 0);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  Serial.println("Gate motor stopped.");
}


// ==================================================
// 9. SELECT TARGET BOX
// ==================================================

String getTargetColor(String material) {

  if (material == "METAL") {
    return "RED";
  }

  if (material == "PLASTIC") {
    return "BLUE";
  }

  if (material == "PAPER") {
    return "PURPLE";
  }

  if (material == "ORGANIC") {
    return "YELLOW";
  }

  return "";
}


// ==================================================
// 10. ROTATE UNTIL CORRECT BOX IS FOUND
// ==================================================

bool findTargetBox() {

  Serial.println("Rotating platform...");

  unsigned long startTime = millis();

  startRotation();

  while (millis() - startTime < MAX_SEARCH_TIME) {

    detectedColor = readBoxColor();

    // Stop immediately when the target is detected.
    if (detectedColor == targetColor) {

      stopRotation();

      Serial.println("Correct box found!");
      Serial.println("Platform stopped.");

      return true;
    }

    delay(100);
  }

  // Safety timeout.
  stopRotation();

  Serial.println("ERROR: Target box not found!");
  Serial.println("Platform stopped for safety.");

  return false;
}


// ==================================================
// 11. SETUP
// ==================================================

void setup() {

  Serial.begin(9600);

  // Set PWM range to 0-255.
  analogWriteResolution(8);

  // TCS3200 pins.
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(SENSOR_OUT, INPUT);

  // TCS3200 frequency scaling: 20%.
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);

  // Motor 1 pins.
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Motor 2 pins.
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Initially stop both motors.
  stopRotation();

  analogWrite(ENB, 0);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  Serial.println();
  Serial.println("==============================");
  Serial.println(" SMART RECYCLING SORTER");
  Serial.println(" STM32F103C8T6");
  Serial.println("==============================");

  Serial.println("Enter material:");
  Serial.println("METAL / PLASTIC / PAPER / ORGANIC");
}


// ==================================================
// 12. MAIN LOOP
// ==================================================

void loop() {

  if (Serial.available() > 0) {

    String material = Serial.readStringUntil('\n');

    material.trim();
    material.toUpperCase();

    targetColor = getTargetColor(material);

    // Validate material.
    if (targetColor == "") {

      Serial.println("Invalid material!");
      Serial.println("Try again.");

      return;
    }

    Serial.println();

    Serial.print("Selected material: ");
    Serial.println(material);

    Serial.print("Target box color: ");
    Serial.println(targetColor);

    // MOTOR 1: Find the correct recycling box.
    bool boxFound = findTargetBox();

    if (boxFound) {

      // MOTOR 2: Open the gate.
      openGate();

      Serial.println("Sorting operation completed!");

    } else {

      Serial.println("Sorting operation cancelled.");
    }

    Serial.println("------------------------------");
    Serial.println("Ready for the next item.");
    Serial.println("Enter material:");
  }
}
