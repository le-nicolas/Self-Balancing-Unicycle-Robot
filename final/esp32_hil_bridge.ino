#include <ArduinoJson.h>
#include <SimpleFOC.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// =========================
// WiFi and UDP
// =========================
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* PC_IP = "192.168.1.100";
const int PC_PORT = 5005;
const int LOCAL_PORT = 5006;

// =========================
// Timing
// =========================
static const uint32_t LOOP_HZ = 250;
static const uint32_t LOOP_US = 1000000u / LOOP_HZ;
static const uint32_t CMD_TIMEOUT_US = 150000u;

// =========================
// BMI088
// =========================
static const uint8_t BMI088_ACCEL_ADDR = 0x19;
static const uint8_t BMI088_GYRO_ADDR = 0x69;

// =========================
// Hardware pins
// =========================
// I2C
static const int I2C_SDA_PIN = 21;
static const int I2C_SCL_PIN = 22;

// DRV8313 (3PWM + enable)
static const int DRV_EN = 25;
static const int DRV_IN1 = 26;
static const int DRV_IN2 = 27;
static const int DRV_IN3 = 14;

// BTS7960 (drive wheel)
static const int BTS_RPWM = 32;
static const int BTS_LPWM = 33;
static const int BTS_REN = 18;
static const int BTS_LEN = 19;

// =========================
// SimpleFOC settings
// =========================
static const int RW_POLE_PAIRS = 7;
static const float RW_SUPPLY_VOLTAGE = 12.0f;
static const float RW_MAX_VOLTAGE = 5.0f;
// Incoming normalized command (-1..1) maps to this voltage gain.
// Increase slowly during bring-up.
static const float RW_CMD_TO_VOLT_GAIN = 5.0f;
static const bool RW_INVERT = false;

// =========================
// Globals
// =========================
WiFiUDP udp;

MagneticSensorI2C rw_sensor = MagneticSensorI2C(AS5600_I2C);
BLDCMotor rw_motor = BLDCMotor(RW_POLE_PAIRS);
BLDCDriver3PWM rw_driver = BLDCDriver3PWM(DRV_IN1, DRV_IN2, DRV_IN3, DRV_EN);

float ax_m_s2 = 0.0f;
float ay_m_s2 = 0.0f;
float az_m_s2 = 0.0f;
float gx_dps = 0.0f;
float gy_dps = 0.0f;
float gz_dps = 0.0f;
float reaction_speed_dps = 0.0f;
float reaction_angle_deg = 0.0f;

float rw_target_voltage = 0.0f;
float cmd_rw_norm = 0.0f;
float cmd_drive_norm = 0.0f;
bool cmd_estop = false;

uint32_t last_loop_us = 0;
uint32_t last_cmd_us = 0;
uint32_t tx_seq = 0;
uint32_t last_debug_ms = 0;

float rw_prev_angle_rad = 0.0f;
bool rw_sensor_initialized = false;

// =========================
// Helpers
// =========================
void i2c_write(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void bmi088_init() {
  i2c_write(BMI088_ACCEL_ADDR, 0x7C, 0x04);  // accel active
  delay(5);
  i2c_write(BMI088_ACCEL_ADDR, 0x40, 0xA8);  // accel ODR/BW

  i2c_write(BMI088_GYRO_ADDR, 0x0F, 0x00);   // +-2000 dps
  i2c_write(BMI088_GYRO_ADDR, 0x10, 0x07);   // 100 Hz
  delay(10);
}

void read_bmi088() {
  uint8_t buf[7] = {0};

  // Accel (dummy byte + 6 data bytes)
  Wire.beginTransmission(BMI088_ACCEL_ADDR);
  Wire.write(0x12);
  Wire.endTransmission(false);
  int accel_n = Wire.requestFrom((int)BMI088_ACCEL_ADDR, 7);
  if (accel_n == 7) {
    Wire.read();  // dummy
    for (int i = 0; i < 6; ++i) {
      buf[i] = Wire.read();
    }
    int16_t raw_ax = (int16_t)((buf[1] << 8) | buf[0]);
    int16_t raw_ay = (int16_t)((buf[3] << 8) | buf[2]);
    int16_t raw_az = (int16_t)((buf[5] << 8) | buf[4]);
    const float accel_scale = 9.81f * 6.0f / 32768.0f;
    ax_m_s2 = raw_ax * accel_scale;
    ay_m_s2 = raw_ay * accel_scale;
    az_m_s2 = raw_az * accel_scale;
  }

  // Gyro (6 bytes)
  Wire.beginTransmission(BMI088_GYRO_ADDR);
  Wire.write(0x02);
  Wire.endTransmission(false);
  int gyro_n = Wire.requestFrom((int)BMI088_GYRO_ADDR, 6);
  if (gyro_n == 6) {
    for (int i = 0; i < 6; ++i) {
      buf[i] = Wire.read();
    }
    int16_t raw_gx = (int16_t)((buf[1] << 8) | buf[0]);
    int16_t raw_gy = (int16_t)((buf[3] << 8) | buf[2]);
    int16_t raw_gz = (int16_t)((buf[5] << 8) | buf[4]);
    const float gyro_scale = 2000.0f / 32768.0f;
    gx_dps = raw_gx * gyro_scale;
    gy_dps = raw_gy * gyro_scale;
    gz_dps = raw_gz * gyro_scale;
  }
}

void update_reaction_sensor() {
  rw_sensor.update();
  float angle_rad = rw_sensor.getAngle();
  if (!rw_sensor_initialized) {
    rw_prev_angle_rad = angle_rad;
    rw_sensor_initialized = true;
  }

  float delta = angle_rad - rw_prev_angle_rad;
  if (delta > PI) delta -= _2PI;
  if (delta < -PI) delta += _2PI;
  rw_prev_angle_rad = angle_rad;

  reaction_angle_deg += delta * 180.0f / PI;
  reaction_speed_dps = rw_sensor.getVelocity() * 180.0f / PI;
}

void receive_command_packet() {
  int packet_size = udp.parsePacket();
  while (packet_size > 0) {
    char buf[192];
    int n = udp.read(buf, sizeof(buf) - 1);
    if (n > 0) {
      buf[n] = 0;
      StaticJsonDocument<192> doc;
      if (deserializeJson(doc, buf) == DeserializationError::Ok) {
        cmd_rw_norm = doc["rt"] | 0.0f;
        cmd_drive_norm = doc["dt"] | 0.0f;
        cmd_estop = (doc["estop"] | 0) != 0;
        cmd_rw_norm = constrain(cmd_rw_norm, -1.0f, 1.0f);
        cmd_drive_norm = constrain(cmd_drive_norm, -1.0f, 1.0f);
        last_cmd_us = micros();
      }
    }
    packet_size = udp.parsePacket();
  }
}

void apply_drive_command(float drive_norm, bool estop_active) {
  if (estop_active) {
    drive_norm = 0.0f;
  }
  drive_norm = constrain(drive_norm, -1.0f, 1.0f);
  int pwm = (int)(fabsf(drive_norm) * 255.0f);
  if (drive_norm >= 0.0f) {
    ledcWrite(0, pwm);
    ledcWrite(1, 0);
  } else {
    ledcWrite(0, 0);
    ledcWrite(1, pwm);
  }
}

void send_sensor_packet(uint32_t ts_us) {
  StaticJsonDocument<384> doc;
  doc["ax"] = ax_m_s2;
  doc["ay"] = ay_m_s2;
  doc["az"] = az_m_s2;
  doc["gx"] = gx_dps;
  doc["gy"] = gy_dps;
  doc["gz"] = gz_dps;
  doc["reaction_speed"] = reaction_speed_dps;
  doc["reaction_angle"] = reaction_angle_deg;
  doc["wheel_angle"] = 0.0f;
  doc["ts"] = ts_us;
  doc["seq"] = tx_seq++;

  char out[384];
  size_t len = serializeJson(doc, out);
  udp.beginPacket(PC_IP, PC_PORT);
  udp.write((const uint8_t*)out, len);
  udp.endPacket();
}

void setup_simplefoc() {
  rw_sensor.init(&Wire);
  rw_driver.voltage_power_supply = RW_SUPPLY_VOLTAGE;
  rw_driver.pwm_frequency = 25000;
  rw_driver.init();

  rw_motor.linkSensor(&rw_sensor);
  rw_motor.linkDriver(&rw_driver);
  rw_motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  rw_motor.controller = MotionControlType::torque;
  rw_motor.torque_controller = TorqueControlType::voltage;
  rw_motor.voltage_limit = RW_MAX_VOLTAGE;
  rw_motor.init();
  rw_motor.initFOC();
}

// =========================
// Setup / Loop
// =========================
void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);
  bmi088_init();
  setup_simplefoc();

  pinMode(BTS_REN, OUTPUT);
  pinMode(BTS_LEN, OUTPUT);
  digitalWrite(BTS_REN, HIGH);
  digitalWrite(BTS_LEN, HIGH);
  ledcSetup(0, 20000, 8);
  ledcAttachPin(BTS_RPWM, 0);
  ledcSetup(1, 20000, 8);
  ledcAttachPin(BTS_LPWM, 1);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected: ");
  Serial.println(WiFi.localIP());
  udp.begin(LOCAL_PORT);

  uint32_t now_us = micros();
  last_loop_us = now_us;
  last_cmd_us = now_us;
  last_debug_ms = millis();

  Serial.println("ESP32 HIL bridge ready");
}

void loop() {
  // High-frequency FOC updates for smooth torque behavior.
  rw_motor.loopFOC();
  rw_motor.move(rw_target_voltage);

  // Non-blocking command receive every cycle.
  receive_command_packet();

  uint32_t now_us = micros();
  bool comm_timeout = ((uint32_t)(now_us - last_cmd_us) > CMD_TIMEOUT_US);
  bool estop_active = cmd_estop || comm_timeout;

  // Command mapping: normalized [-1,1] -> motor voltage.
  float rw_norm = estop_active ? 0.0f : cmd_rw_norm;
  if (RW_INVERT) rw_norm = -rw_norm;
  rw_target_voltage = constrain(rw_norm * RW_CMD_TO_VOLT_GAIN, -RW_MAX_VOLTAGE, RW_MAX_VOLTAGE);
  apply_drive_command(cmd_drive_norm, estop_active);

  // Sensor/network loop at fixed rate.
  if ((uint32_t)(now_us - last_loop_us) >= LOOP_US) {
    last_loop_us += LOOP_US;
    read_bmi088();
    update_reaction_sensor();
    send_sensor_packet(now_us);
  }

  uint32_t now_ms = millis();
  if (now_ms - last_debug_ms >= 1000u) {
    last_debug_ms = now_ms;
    Serial.print("rw_norm=");
    Serial.print(rw_norm, 3);
    Serial.print(" rw_v=");
    Serial.print(rw_target_voltage, 3);
    Serial.print(" drive=");
    Serial.print(cmd_drive_norm, 3);
    Serial.print(" speed_dps=");
    Serial.print(reaction_speed_dps, 1);
    Serial.print(" estop=");
    Serial.println(estop_active ? 1 : 0);
  }
}
