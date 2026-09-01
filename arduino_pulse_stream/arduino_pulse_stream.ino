/*
  Pulse Block Streaming XYZ-angle + Servo

  Python membentuk timeline pulsa X/Y dan target sudut Z dari kecepatan lintasan
  mm/s. Firmware mengeksekusi X/Y dengan DDA dan Z dengan umpan balik AS5600.
  Servo memakai paket CRC + sequence agar retry tidak menggerakkan servo dua kali.
*/

#include <avr/interrupt.h>
#include <util/atomic.h>
#include <Wire.h>

const byte PIN_STEP_X = 8;  const byte PIN_DIR_X = 9;
const byte PIN_STEP_Y = 6;  const byte PIN_DIR_Y = 7;
const byte PIN_STEP_Z = 4;  const byte PIN_DIR_Z = 5;
// Arah mekanis sumbu Y dibalik terhadap level DIR lama.  Satu konstanta ini
// dipakai oleh gerak streaming dan homing agar arah fisik Y selalu konsisten.
const bool INVERT_Y_DIRECTION = true;
const byte PIN_SERVO_GRIP = 2;
const byte PIN_AS5600_POWER = 3;
const byte PIN_LIMIT_X = 11; const byte PIN_LIMIT_Y = 12; const byte PIN_LIMIT_Z = 13;
const byte PIN_BUTTON_HOME = A1; const byte PIN_BUTTON_RUN = A2;

const byte SLOT_COUNT = 32;
const byte MAX_SLICES = 2;
const unsigned int ISR_TICK_US = 20;
const unsigned int XY_STEP_PULSE_HIGH_US = 2;
// Z HIGH mengikuti satu periode Timer1, yaitu 20 us.
const unsigned int Z_STEP_PULSE_HIGH_US = ISR_TICK_US;
const unsigned long HOMING_TIMEOUT_MS = 90000UL;
const unsigned int HOME_X_HALF_PERIOD_US = 30;
const unsigned int HOME_Y_HALF_PERIOD_US = 30;
const unsigned int HOME_Z_HALF_PERIOD_US = 20;
const unsigned int RX_BUFFER_SIZE = 128;
const unsigned int SERVO_PERIOD_US = 10000;
const unsigned int SERVO_MIN_PULSE_US = 544;
const unsigned int SERVO_MAX_PULSE_US = 2400;
// Servo GRIP diberi pulsa PWM selama 1 detik penuh ketika aksi GRIP baru
// dimulai.  Setelah itu sinyal dihentikan agar aksi tidak mengganggu
// streaming pulsa X/Y/Z.  RELEASE memakai durasi yang sama agar mekanisme
// pelepasan tetap memiliki waktu mekanis yang cukup.
const unsigned long SERVO_GRIP_ACTIVE_MS = 1000UL;
const unsigned long SERVO_RELEASE_ACTIVE_MS = 1000UL;
const byte SERVO_GRIP_ANGLE = 35;
const byte SERVO_RELEASE_ANGLE = 100;
const byte AS5600_ADDRESS = 0x36;
const float MOTOR_STEPS_PER_REV = 6400.0;
// AS5600 dipasang pada poros OUTPUT gearbox. Karena itu sudut feedback
// dihitung langsung dari raw AS5600 (tidak pernah dikalikan rasio gearbox).
// Rasio hanya dipakai saat mengubah sudut output menjadi pulsa MOTOR.
const float Z_MOTOR_TO_OUTPUT_REDUCTION = 20.0;
const float Z_MOTOR_STEPS_PER_OUTPUT_DEG =
  (MOTOR_STEPS_PER_REV * Z_MOTOR_TO_OUTPUT_REDUCTION) / 360.0;
const int Z_TOLERANCE_DECI_DEG = 5; // 0,5 derajat
// AS5600 dibaca 20 ms sekali. Generator pulsa Z tetap berjalan pada ISR 20 us.
const unsigned long Z_ENCODER_SAMPLE_US = 20000UL;
const int Z_ENCODER_MAX_DELTA_COUNTS = 342; // sekitar 30 derajat
const unsigned long Z_STALL_TIMEOUT_MS = 3000UL;
const unsigned long Z_DIRECTION_CHECK_MS = 250UL;
const int Z_DIRECTION_MIN_REVERSE_DECI_DEG = 2; // 0,2 derajat

enum SlotState : byte { SLOT_FREE = 0, SLOT_LOADING = 1, SLOT_READY = 2 };

struct MotionSlot {
  // Sequence 32-bit agar trajektori lebih dari 65.535 blok tidak kembali
  // ke sequence 0 dan menimpa slot ring yang masih aktif.
  unsigned long sequence;
  byte count;
  int16_t dx[MAX_SLICES];
  int16_t dy[MAX_SLICES];
  int16_t zTarget[MAX_SLICES];
};

MotionSlot slots[SLOT_COUNT];
volatile byte slotState[SLOT_COUNT] = {SLOT_FREE};

char rxBuffer[RX_BUFFER_SIZE];
unsigned int rxLength = 0;
bool rxOverflow = false;

volatile bool streamRunning = false;
volatile bool streamConfigured = false;
volatile bool streamDonePending = false;
volatile bool bufferWaiting = false;
volatile bool bufferWaitPending = false;
volatile unsigned long activeSequence = 0;
volatile byte activeSlice = 0;
volatile unsigned long totalBlocks = 0;
volatile unsigned int ticksPerSlice = 500;
volatile unsigned int tickInSlice = 500;
volatile unsigned int absX = 0, absY = 0;
volatile unsigned int phaseX = 0, phaseY = 0;
volatile int8_t signX = 0, signY = 0;
volatile long positionX = 0, positionY = 0;
volatile int16_t zTargetDeciDeg = 0;
volatile bool zPulseEnabled = false;
volatile bool zStepPinHigh = false;
volatile unsigned int zPulseIntervalTicks = 1;
volatile unsigned int zPulseCounter = 0;
volatile bool xyStreamFinished = false;
// Sequence DONE tidak lagi disimpan hanya pada satu variabel int16. Main loop
// melaporkan seluruh rentang sequence yang sudah dilewati ISR, sehingga ACK
// tidak hilang ketika beberapa blok selesai sebelum serviceEvents berjalan.
unsigned long reportedDoneSequence = 0;

long zTrackedCounts = 0;
long zCurrentDeciDeg = 0;
unsigned int zLastRaw = 0;
bool zEncoderInitialized = false;
bool zEncoderHealthy = false;
bool zHomed = false;
byte zEncoderRejectCount = 0;
int8_t zEncoderDirectionSign = 1;
unsigned long lastZEncoderSampleUs = 0;
float zDegPerCm = 15.0;
float zSpeedMmS = 20.0;
unsigned int zConfiguredPulseIntervalTicks = 5;
bool zFaultPending = false;
bool zFaultLatched = false;
byte zReadFailureCount = 0;
bool zProgressTracking = false;
long zProgressReferenceCounts = 0;
unsigned long zLastProgressMs = 0;
int8_t zCommandDirection = 0;
long zDirectionReferenceDeciDeg = 0;
unsigned long zDirectionCheckStartedMs = 0;
bool homingActive = false;
bool homingAbortRequested = false;

void readSerialDuringHoming();

bool lastHomeButton = HIGH, lastRunButton = HIGH;
unsigned long homeChangedMs = 0, runChangedMs = 0;

bool servoActive = false;
bool servoPinHigh = false;
bool bottleGripped = false;
byte servoAngle = SERVO_RELEASE_ANGLE;
unsigned int servoPulseUs = 1500;
unsigned long servoStartedMs = 0;
unsigned long servoActiveDurationMs = SERVO_RELEASE_ACTIVE_MS;
unsigned long servoPeriodStartedUs = 0;
unsigned long servoHighStartedUs = 0;
bool lastServoValid = false;
bool lastServoFinished = false;
unsigned int lastServoSequence = 0;
bool lastServoGrip = false;
byte lastServoAngle = SERVO_RELEASE_ANGLE;

void printReady() {
  Serial.println(F("PULSE_BLOCK_READY:32:SEQ_BITS:32:BLOCK_SLICES:2:SLOTS:32:TICK_US:20:AXES:XYZ_ANGLE:Y_DIR_INVERTED:1:Z_HIGH_US:20:Z_AS5600:1:Z_ENCODER_OUTPUT:1:Z_SAMPLE_MS:20:REDUCTION:20:SERVO_FIXED:1:SERVO_GRIP_MS:1000:STATE_RECOVERY:1"));
}

uint16_t crc16Ccitt(const char *text) {
  uint16_t crc = 0xFFFF;
  while (*text) {
    crc ^= (uint16_t)(uint8_t)(*text++) << 8;
    for (byte bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

void setPulseTimerInterrupt(bool enabled) {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    if (enabled) TIMSK1 |= _BV(OCIE1A);
    else {
      TIMSK1 &= ~_BV(OCIE1A);
      PORTB &= ~_BV(PB0);
      PORTD &= ~(_BV(PD6) | _BV(PD4));
      zStepPinHigh = false;
    }
  }
}

void clearStepPinsISR() {
  PORTB &= ~_BV(PB0); // D8 STEP X
  PORTD &= ~(_BV(PD6) | _BV(PD4)); // D6 STEP Y, D4 STEP Z
  zStepPinHigh = false;
}

void clearXYStepPinsISR() {
  PORTB &= ~_BV(PB0);
  PORTD &= ~_BV(PD6);
}

void finishXYStepPulseISR(bool emitted) {
  if (!emitted) return;
  delayMicroseconds(XY_STEP_PULSE_HIGH_US);
  clearXYStepPinsISR();
}

void setDirectionsISR(int16_t dx, int16_t dy) {
  if (dx >= 0) PORTB &= ~_BV(PB1); else PORTB |= _BV(PB1);
  bool yDirectionHigh = dy < 0;
  if (INVERT_Y_DIRECTION) yDirectionHigh = !yDirectionHigh;
  if (yDirectionHigh) PORTD |= _BV(PD7); else PORTD &= ~_BV(PD7);
}

bool loadSliceISR() {
  if (activeSequence >= totalBlocks) {
    streamRunning = false;
    xyStreamFinished = true;
    return false;
  }

  byte slotIndex = activeSequence % SLOT_COUNT;
  if (slotState[slotIndex] != SLOT_READY || slots[slotIndex].sequence != activeSequence) {
    streamRunning = false;
    bufferWaiting = true;
    bufferWaitPending = true;
    TIMSK1 &= ~_BV(OCIE1A);
    return false;
  }

  if (activeSlice >= slots[slotIndex].count) {
    slotState[slotIndex] = SLOT_FREE;
    activeSequence++;
    activeSlice = 0;
    if (activeSequence >= totalBlocks) {
      streamRunning = false;
      xyStreamFinished = true;
      return false;
    }
    slotIndex = activeSequence % SLOT_COUNT;
    if (slotState[slotIndex] != SLOT_READY || slots[slotIndex].sequence != activeSequence) {
      streamRunning = false;
      bufferWaiting = true;
      bufferWaitPending = true;
      TIMSK1 &= ~_BV(OCIE1A);
      return false;
    }
  }

  int16_t dx = slots[slotIndex].dx[activeSlice];
  int16_t dy = slots[slotIndex].dy[activeSlice];
  int16_t targetZ = slots[slotIndex].zTarget[activeSlice];
  activeSlice++;
  setDirectionsISR(dx, dy);
  signX = (dx > 0) - (dx < 0);
  signY = (dy > 0) - (dy < 0);
  absX = abs(dx); absY = abs(dy);
  phaseX = phaseY = 0;
  zTargetDeciDeg = targetZ;
  tickInSlice = 0;
  return true;
}

ISR(TIMER1_COMPA_vect) {
  clearXYStepPinsISR();
  // STEP Z dipertahankan HIGH selama Z_STEP_PULSE_HIGH_US = 20 us.
  // Kecepatan Z ditentukan oleh interval antarpulsa.
  if (zStepPinHigh) {
    PORTD &= ~_BV(PD4);
    zStepPinHigh = false;
  }
  bool emittedXY = false;
  if (zPulseEnabled) {
    zPulseCounter++;
    if (zPulseCounter >= zPulseIntervalTicks) {
      zPulseCounter = 0;
      PORTD |= _BV(PD4);
      zStepPinHigh = true;
    }
  } else {
    zPulseCounter = 0;
  }

  if (!streamRunning) {
    finishXYStepPulseISR(emittedXY);
    return;
  }
  if (tickInSlice >= ticksPerSlice && !loadSliceISR()) {
    finishXYStepPulseISR(emittedXY);
    return;
  }
  phaseX += absX;
  if (phaseX >= ticksPerSlice) {
    phaseX -= ticksPerSlice;
    PORTB |= _BV(PB0);
    positionX += signX;
    emittedXY = true;
  }
  phaseY += absY;
  if (phaseY >= ticksPerSlice) {
    phaseY -= ticksPerSlice;
    PORTD |= _BV(PD6);
    positionY += signY;
    emittedXY = true;
  }
  tickInSlice++;
  finishXYStepPulseISR(emittedXY);
}

void configureTimer1() {
  cli();
  TCCR1A = 0;
  TCCR1B = _BV(WGM12) | _BV(CS11);
  OCR1A = (unsigned int)(ISR_TICK_US * 2UL - 1UL);
  TIMSK1 = 0;
  sei();
}

void resetStream() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    streamRunning = false;
    streamConfigured = false;
    streamDonePending = false;
    bufferWaiting = false;
    bufferWaitPending = false;
    xyStreamFinished = false;
    zPulseEnabled = false;
    zPulseCounter = 0;
    zTargetDeciDeg = (int16_t)zCurrentDeciDeg;
    zCommandDirection = 0;
    zDirectionReferenceDeciDeg = zCurrentDeciDeg;
    zDirectionCheckStartedMs = 0;
    activeSequence = 0;
    activeSlice = 0;
    totalBlocks = 0;
    tickInSlice = ticksPerSlice;
    reportedDoneSequence = 0;
    clearStepPinsISR();
    for (byte i = 0; i < SLOT_COUNT; i++) slotState[i] = SLOT_FREE;
  }
}

unsigned int servoAngleToPulseUs(byte angle) {
  return SERVO_MIN_PULSE_US +
    (((unsigned long)(SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US) * angle) / 180UL);
}

void stopServo() {
  servoPinHigh = false;
  servoActive = false;
  digitalWrite(PIN_SERVO_GRIP, LOW);
}

void printServoActionAck() {
  Serial.print(F("ACTION_ACK:")); Serial.print(lastServoSequence);
  Serial.print(F(":")); Serial.print(lastServoGrip ? F("GRIP") : F("RELEASE"));
  Serial.print(F(":")); Serial.println(lastServoAngle);
}

void printServoFinished() {
  Serial.println(lastServoGrip ? F("GOK") : F("ROK"));
}

void startServoAction(unsigned int sequence, bool grip, byte angle) {
  // Perintah GRIP baru hanya menggerakkan servo saat status sebelumnya belum
  // menggenggam. Dengan demikian retry atau checkpoint GRIP ganda tidak
  // memperpanjang/menjalankan ulang aktuasi satu detik.
  bool gripAlreadyActive = grip && bottleGripped;
  lastServoValid = true;
  lastServoFinished = false;
  lastServoSequence = sequence;
  lastServoGrip = grip;
  lastServoAngle = angle;
  bottleGripped = grip;
  servoAngle = angle;
  servoPulseUs = servoAngleToPulseUs(angle);
  servoActiveDurationMs = grip ? SERVO_GRIP_ACTIVE_MS : SERVO_RELEASE_ACTIVE_MS;
  printServoActionAck();
  if (gripAlreadyActive) {
    stopServo();
    lastServoFinished = true;
    printServoFinished();
    return;
  }
  servoStartedMs = millis();
  servoPeriodStartedUs = micros() - SERVO_PERIOD_US;
  servoPinHigh = false;
  servoActive = true;
}

void serviceServo() {
  if (!servoActive) return;
  unsigned long nowUs = micros();
  if (!servoPinHigh && (unsigned long)(nowUs - servoPeriodStartedUs) >= SERVO_PERIOD_US) {
    servoPeriodStartedUs = nowUs;
    servoHighStartedUs = nowUs;
    digitalWrite(PIN_SERVO_GRIP, HIGH);
    servoPinHigh = true;
  }
  if (servoPinHigh && (unsigned long)(nowUs - servoHighStartedUs) >= servoPulseUs) {
    digitalWrite(PIN_SERVO_GRIP, LOW);
    servoPinHigh = false;
  }
  if ((unsigned long)(millis() - servoStartedMs) >= servoActiveDurationMs) {
    stopServo();
    lastServoFinished = true;
    printServoFinished();
  }
}

bool readAS5600Raw(unsigned int &raw) {
  for (byte attempt = 0; attempt < 3; attempt++) {
    Wire.beginTransmission(AS5600_ADDRESS);
    Wire.write(0x0C);
    if (Wire.endTransmission(false) == 0 &&
        Wire.requestFrom(AS5600_ADDRESS, (byte)2) == 2) {
      raw = (((unsigned int)Wire.read() << 8) | Wire.read()) & 0x0FFF;
      return true;
    }
    delayMicroseconds(10);
  }
  return false;
}

bool updateZEncoder() {
  unsigned int raw;
  if (!readAS5600Raw(raw)) {
    if (zReadFailureCount < 255) zReadFailureCount++;
    if (zReadFailureCount >= 3) {
      zEncoderHealthy = false;
      if (!zFaultLatched) {
        zFaultLatched = true;
        zFaultPending = true;
      }
    }
    return false;
  }
  zReadFailureCount = 0;
  zFaultLatched = false;
  zEncoderHealthy = true;
  if (!zEncoderInitialized) {
    zLastRaw = raw;
    zTrackedCounts = 0;
    zCurrentDeciDeg = 0;
    zEncoderInitialized = true;
    return true;
  }
  int delta = (int)raw - (int)zLastRaw;
  if (delta > 2048) delta -= 4096;
  else if (delta < -2048) delta += 4096;
  if (abs(delta) > Z_ENCODER_MAX_DELTA_COUNTS) {
    zEncoderRejectCount++;
    if (zEncoderRejectCount >= 3) {
      zLastRaw = raw;
      zEncoderRejectCount = 0;
      zEncoderHealthy = false;
      zFaultPending = true;
      return false;
    }
    return true;
  }
  zEncoderRejectCount = 0;
  zTrackedCounts += (long)delta * (long)zEncoderDirectionSign;
  zLastRaw = raw;
  // Nilai ini adalah sudut poros OUTPUT gearbox. Jangan kalikan dengan
  // Z_MOTOR_TO_OUTPUT_REDUCTION; rasio sudah hanya digunakan untuk pulsa.
  zCurrentDeciDeg = (zTrackedCounts * 3600L) / 4096L;
  return true;
}

void zeroZEncoderTracking() {
  unsigned int raw;
  if (readAS5600Raw(raw)) {
    zLastRaw = raw;
    zTrackedCounts = 0;
    zCurrentDeciDeg = 0;
    zEncoderInitialized = true;
    zEncoderHealthy = true;
    zEncoderRejectCount = 0;
  } else {
    zEncoderHealthy = false;
    zFaultPending = true;
  }
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    zTargetDeciDeg = 0;
    zPulseEnabled = false;
    zPulseCounter = 0;
  }
}

bool calibrateZEncoderDirectionAtHome() {
  zEncoderDirectionSign = 1;
  zeroZEncoderTracking();
  if (!zEncoderHealthy) return false;

  digitalWrite(PIN_DIR_Z, HIGH);
  delayMicroseconds(5);
  bool progressDetected = false;
  for (unsigned int step = 0; step < 2000; step++) {
    digitalWrite(PIN_STEP_Z, HIGH);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
    digitalWrite(PIN_STEP_Z, LOW);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
    if ((step & 0x07) == 0) {
      readSerialDuringHoming();
      if (homingAbortRequested) return false;
      lastZEncoderSampleUs = 0;
      if (!updateZEncoder()) return false;
      if (labs(zCurrentDeciDeg) >= 5) { // minimal perubahan 0,5°
        progressDetected = true;
        break;
      }
    }
  }
  if (!progressDetected) return false;
  zEncoderDirectionSign = zCurrentDeciDeg < 0 ? -1 : 1;

  digitalWrite(PIN_DIR_Z, LOW);
  delayMicroseconds(5);
  unsigned long guard = 0;
  while (digitalRead(PIN_LIMIT_Z) != LOW && guard < 5000UL) {
    digitalWrite(PIN_STEP_Z, HIGH);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
    digitalWrite(PIN_STEP_Z, LOW);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
    guard++;
    if ((guard & 0x3F) == 0) {
      readSerialDuringHoming();
      if (homingAbortRequested) return false;
    }
  }
  if (digitalRead(PIN_LIMIT_Z) != LOW) return false;
  zeroZEncoderTracking();
  return zEncoderHealthy;
}

void updateZPulseRateConfig() {
  float zDegPerMm = zDegPerCm / 10.0;
  // Kecepatan linear -> sudut OUTPUT -> pulsa MOTOR. Ini bukan konversi
  // sudut encoder; feedback tetap berada pada satuan sudut output.
  float pulseHz = zSpeedMmS * zDegPerMm * Z_MOTOR_STEPS_PER_OUTPUT_DEG;
  if (pulseHz < 1.0) pulseHz = 1.0;
  float timerHz = 1000000.0 / ISR_TICK_US;
  if (pulseHz > timerHz) pulseHz = timerHz;
  unsigned long interval = (unsigned long)(timerHz / pulseHz + 0.5);
  // HIGH Z harus 20 us dan LOW minimal 20 us, sehingga interval minimum dua
  // tick (periode 40 us, maksimum 25 kHz).
  if (interval < 2) interval = 2;
  if (interval > 65535UL) interval = 65535UL;
  zConfiguredPulseIntervalTicks = (unsigned int)interval;
}

void serviceZClosedLoop() {
  unsigned long nowUs = micros();
  if ((unsigned long)(nowUs - lastZEncoderSampleUs) < Z_ENCODER_SAMPLE_US) return;
  lastZEncoderSampleUs = nowUs;
  if (!updateZEncoder()) {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { zPulseEnabled = false; }
    return;
  }

  int target;
  bool finalWait, streamIsWaiting;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    target = zTargetDeciDeg;
    finalWait = xyStreamFinished;
    streamIsWaiting = bufferWaiting;
  }
  if (streamIsWaiting) {
    // Timer1 memang dihentikan selama BUFFER_WAIT. Jangan menganggap
    // tidak adanya langkah selama host mengisi buffer sebagai Z_STALL.
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      zPulseEnabled = false;
      zPulseCounter = 0;
    }
    zProgressTracking = false;
    zDirectionReferenceDeciDeg = zCurrentDeciDeg;
    zDirectionCheckStartedMs = millis();
    return;
  }
  long error = (long)target - zCurrentDeciDeg;
  bool withinTolerance = labs(error) <= Z_TOLERANCE_DECI_DEG;

  if (!withinTolerance) {
    unsigned long nowMs = millis();
    int8_t commandDirection = error > 0 ? 1 : -1;
    int8_t previousDirection = zCommandDirection;
    bool directionChanged = commandDirection != previousDirection;
    if (directionChanged) {
      // Mulai pemeriksaan baru setiap kali arah target berubah. Target boleh
      // berubah setiap slice; yang penting tanda error tetap konsisten.
      zCommandDirection = commandDirection;
      zDirectionReferenceDeciDeg = zCurrentDeciDeg;
      zDirectionCheckStartedMs = nowMs;
    } else if (zDirectionCheckStartedMs != 0 &&
               (unsigned long)(nowMs - zDirectionCheckStartedMs) >= Z_DIRECTION_CHECK_MS) {
      long directedProgress = (zCurrentDeciDeg - zDirectionReferenceDeciDeg) * commandDirection;
      if (directedProgress < -Z_DIRECTION_MIN_REVERSE_DECI_DEG) {
        resetStream();
        setPulseTimerInterrupt(false);
        zProgressTracking = false;
        zCommandDirection = 0;
        Serial.print(F("ERROR:Z_WRONG_DIRECTION:TARGET_DECI=")); Serial.print(target);
        Serial.print(F(":ACTUAL_DECI=")); Serial.print(zCurrentDeciDeg);
        Serial.print(F(":ENC_SIGN=")); Serial.println(zEncoderDirectionSign);
        return;
      }
      zDirectionReferenceDeciDeg = zCurrentDeciDeg;
      zDirectionCheckStartedMs = nowMs;
    }
    if (!zProgressTracking) {
      zProgressTracking = true;
      zProgressReferenceCounts = zTrackedCounts;
      zLastProgressMs = nowMs;
    } else if (labs(zTrackedCounts - zProgressReferenceCounts) >= 2) {
      zProgressReferenceCounts = zTrackedCounts;
      zLastProgressMs = nowMs;
    } else if ((unsigned long)(nowMs - zLastProgressMs) >= Z_STALL_TIMEOUT_MS) {
      resetStream();
      setPulseTimerInterrupt(false);
      zProgressTracking = false;
      Serial.print(F("ERROR:Z_STALL:TARGET_DECI=")); Serial.print(target);
      Serial.print(F(":ACTUAL_DECI=")); Serial.println(zCurrentDeciDeg);
      return;
    }
    bool positive = error > 0;
    if (!positive && digitalRead(PIN_LIMIT_Z) == LOW) {
      zeroZEncoderTracking();
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { zPulseEnabled = false; zPulseCounter = 0; }
      withinTolerance = target <= Z_TOLERANCE_DECI_DEG;
    } else {
      bool pulseCurrentlyEnabled;
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { pulseCurrentlyEnabled = zPulseEnabled; }
      // Jangan mematikan dan menyalakan ulang pulsa pada setiap sampel
      // encoder. Selama arah masih sama, phase counter ISR dibiarkan berjalan
      // kontinu sehingga tidak muncul jeda 20 ms yang terasa tersendat.
      if (directionChanged || !pulseCurrentlyEnabled) {
        if (directionChanged) {
          ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
            zPulseEnabled = false;
            zPulseCounter = 0;
          }
        }
        digitalWrite(PIN_DIR_Z, positive ? HIGH : LOW);
        if (directionChanged) delayMicroseconds(5);
      }
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        zPulseIntervalTicks = zConfiguredPulseIntervalTicks;
        if (directionChanged || !pulseCurrentlyEnabled) zPulseCounter = 0;
        zPulseEnabled = true;
        TIMSK1 |= _BV(OCIE1A);
      }
    }
  }

  if (withinTolerance) {
    zProgressTracking = false;
    zCommandDirection = 0;
    zDirectionCheckStartedMs = 0;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { zPulseEnabled = false; zPulseCounter = 0; }
    if (finalWait) {
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        xyStreamFinished = false;
        streamDonePending = true;
        if (!streamRunning && !bufferWaiting) TIMSK1 &= ~_BV(OCIE1A);
      }
    }
  }
}

byte readySlotCount() {
  byte ready = 0;
  for (byte i = 0; i < SLOT_COUNT; i++) if (slotState[i] == SLOT_READY) ready++;
  return ready;
}

void printState(bool statusPrefix) {
  bool stateRun, stateXYRun, stateXYWait, stateConfigured;
  unsigned long stateActive, stateTotal;
  long stateX, stateY;
  int stateZDeci, stateZTarget, stateZDirection;
  bool stateZEncoder, stateZHome, stateHoming;
  bool stateGrip, stateServo;
  byte stateAngle;
  bool stateServoValid;
  unsigned int stateServoSequence;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    stateRun = streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled;
    stateXYRun = streamRunning;
    stateXYWait = bufferWaiting;
    stateActive = activeSequence;
    stateTotal = totalBlocks;
    stateConfigured = streamConfigured;
    stateX = positionX;
    stateY = positionY;
    stateZDeci = zCurrentDeciDeg;
    stateZTarget = zTargetDeciDeg;
    stateZDirection = zCommandDirection;
    stateZEncoder = zEncoderHealthy;
    stateZHome = zHomed;
    stateHoming = homingActive;
    stateGrip = bottleGripped;
    stateAngle = servoAngle;
    stateServo = servoActive;
    stateServoValid = lastServoValid;
    stateServoSequence = lastServoSequence;
  }
  Serial.print(statusPrefix ? F("STATUS:RUN=") : F("STATE:RUN="));
  Serial.print(stateRun ? 1 : 0);
  Serial.print(F(":XYRUN=")); Serial.print(stateXYRun ? 1 : 0);
  Serial.print(F(":XYWAIT=")); Serial.print(stateXYWait ? 1 : 0);
  Serial.print(F(":ACTIVE=")); Serial.print(stateActive);
  Serial.print(F(":READY=")); Serial.print(readySlotCount());
  Serial.print(F(":SLOTS=")); Serial.print(SLOT_COUNT);
  Serial.print(F(":TOTAL=")); Serial.print(stateTotal);
  Serial.print(F(":CFG=")); Serial.print(stateConfigured ? 1 : 0);
  Serial.print(F(":X=")); Serial.print(stateX);
  Serial.print(F(":Y=")); Serial.print(stateY);
  Serial.print(F(":ZDECI=")); Serial.print(stateZDeci);
  Serial.print(F(":ZTGT=")); Serial.print(stateZTarget);
  Serial.print(F(":ZERR=")); Serial.print((long)stateZTarget - stateZDeci);
  Serial.print(F(":ZDIR=")); Serial.print(stateZDirection);
  Serial.print(F(":ZBUSY=")); Serial.print((labs((long)stateZTarget - stateZDeci) > Z_TOLERANCE_DECI_DEG) ? 1 : 0);
  Serial.print(F(":ZENC=")); Serial.print(stateZEncoder ? 1 : 0);
  Serial.print(F(":HOME=")); Serial.print(stateZHome ? 1 : 0);
  Serial.print(F(":HOMING=")); Serial.print(stateHoming ? 1 : 0);
  Serial.print(F(":GRIP=")); Serial.print(stateGrip ? 1 : 0);
  Serial.print(F(":ANGLE=")); Serial.print(stateAngle);
  Serial.print(F(":SERVO=")); Serial.print(stateServo ? 1 : 0);
  Serial.print(F(":SSEQ="));
  if (stateServoValid) Serial.println(stateServoSequence);
  else Serial.println(-1);
}

// Homing tetap blocking terhadap pembangkitan pulsa, tetapi tidak lagi membuat
// port serial tampak mati. Selama homing hanya perintah observasi dan STOP yang
// diterima; perintah gerak lain ditolak agar tidak terjadi gerakan bersamaan.
void readSerialDuringHoming() {
  while (Serial.available()) {
    char value = (char)Serial.read();
    if (value == '\r') continue;
    if (value == '\n') {
      if (!rxOverflow && rxLength) {
        rxBuffer[rxLength] = '\0';
        if (!strcmp(rxBuffer, "PING")) Serial.println(F("PONG"));
        else if (!strcmp(rxBuffer, "STATE?")) printState(false);
        else if (!strcmp(rxBuffer, "STATUS")) printState(true);
        else if (!strcmp(rxBuffer, "HELLO")) printReady();
        else if (!strcmp(rxBuffer, "STOP")) {
          homingAbortRequested = true;
          Serial.println(F("HOMING_STOPPING"));
        } else Serial.println(F("ERROR:HOMING_ACTIVE"));
      } else if (rxOverflow) Serial.println(F("ERROR:RX_OVERFLOW"));
      rxLength = 0;
      rxOverflow = false;
    } else if (!rxOverflow) {
      if (rxLength < RX_BUFFER_SIZE - 1) rxBuffer[rxLength++] = value;
      else rxOverflow = true;
    }
  }
}

void executeHome() {
  if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled || servoActive) {
    Serial.println(F("ERROR:HOME_WHILE_BUSY"));
    return;
  }
  resetStream();
  setPulseTimerInterrupt(false);
  homingActive = true;
  homingAbortRequested = false;
  zHomed = false;
  Serial.println(F("HOMING_START"));
  unsigned long started = millis();
  digitalWrite(PIN_DIR_X, HIGH);
  // Dengan pembalikan arah Y, limit home dicapai pada level LOW.
  digitalWrite(PIN_DIR_Y, INVERT_Y_DIRECTION ? LOW : HIGH);
  bool xDone = digitalRead(PIN_LIMIT_X) == LOW;
  bool yDone = digitalRead(PIN_LIMIT_Y) == LOW;
  while (!(xDone && yDone)) {
    readSerialDuringHoming();
    if (homingAbortRequested) {
      digitalWrite(PIN_STEP_X, LOW); digitalWrite(PIN_STEP_Y, LOW);
      homingActive = false;
      Serial.println(F("HOMING_ABORTED"));
      return;
    }
    if (millis() - started > HOMING_TIMEOUT_MS) {
      homingActive = false;
      Serial.println(F("HOMING_FAILED:TIMEOUT_XY"));
      return;
    }
    // X dan Y dipulsa bersamaan. Karena keduanya memakai 30 us, masing-masing
    // memperoleh HIGH 30 us dan LOW 30 us tanpa waktu sumbu lain ikut
    // memperlambat periode homing.
    if (!xDone) digitalWrite(PIN_STEP_X, HIGH);
    if (!yDone) digitalWrite(PIN_STEP_Y, HIGH);
    delayMicroseconds(min(HOME_X_HALF_PERIOD_US, HOME_Y_HALF_PERIOD_US));
    if (!xDone) digitalWrite(PIN_STEP_X, LOW);
    if (!yDone) digitalWrite(PIN_STEP_Y, LOW);
    delayMicroseconds(min(HOME_X_HALF_PERIOD_US, HOME_Y_HALF_PERIOD_US));
    if (!xDone) xDone = digitalRead(PIN_LIMIT_X) == LOW;
    if (!yDone) yDone = digitalRead(PIN_LIMIT_Y) == LOW;
  }
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { positionX = positionY = 0; }
  digitalWrite(PIN_DIR_Z, LOW);
  while (digitalRead(PIN_LIMIT_Z) != LOW) {
    readSerialDuringHoming();
    if (homingAbortRequested) {
      digitalWrite(PIN_STEP_Z, LOW);
      homingActive = false;
      Serial.println(F("HOMING_ABORTED"));
      return;
    }
    if (millis() - started > HOMING_TIMEOUT_MS) {
      homingActive = false;
      Serial.println(F("HOMING_FAILED:TIMEOUT_Z"));
      return;
    }
    digitalWrite(PIN_STEP_Z, HIGH);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
    digitalWrite(PIN_STEP_Z, LOW);
    delayMicroseconds(HOME_Z_HALF_PERIOD_US);
  }
  if (!calibrateZEncoderDirectionAtHome()) {
    homingActive = false;
    if (homingAbortRequested) {
      Serial.println(F("HOMING_ABORTED"));
      return;
    }
    Serial.println(F("HOMING_FAILED:Z_ENCODER_DIRECTION"));
    return;
  }
  zHomed = true;
  homingActive = false;
  Serial.println(F("HOMING_END"));
}

void printBlockAck(byte slotIndex, bool alreadyDone) {
  Serial.print(F("BLOCK_ACK:")); Serial.print(slots[slotIndex].sequence);
  Serial.print(F(":SLOT:")); Serial.print(slotIndex);
  if (alreadyDone) Serial.print(F(":ALREADY_DONE"));
  Serial.println();
}

void parseBlock(char *line) {
  char *sequenceText = line + 6;
  char *separator1 = strchr(sequenceText, ':');
  if (!separator1) { Serial.println(F("NAK:FORMAT")); return; }
  *separator1 = '\0';
  char *countText = separator1 + 1;
  char *separator2 = strchr(countText, ':');
  if (!separator2) { Serial.println(F("NAK:FORMAT")); return; }
  *separator2 = '\0';
  char *crcText = separator2 + 1;
  char *separator3 = strchr(crcText, ':');
  if (!separator3) { Serial.println(F("NAK:FORMAT")); return; }
  *separator3 = '\0';
  char *payload = separator3 + 1;
  if (!*payload) { Serial.println(F("NAK:FORMAT")); return; }

  unsigned long sequence = strtoul(sequenceText, NULL, 10);
  byte count = (byte)atoi(countText);
  uint16_t expectedCrc = (uint16_t)strtoul(crcText, NULL, 16);
  bool configured;
  unsigned long configuredTotal;
  unsigned long currentActive;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    configured = streamConfigured;
    configuredTotal = totalBlocks;
    currentActive = activeSequence;
  }
  if (!configured || sequence >= configuredTotal) { Serial.println(F("NAK:STATE")); return; }
  if (count < 1 || count > MAX_SLICES) { Serial.println(F("NAK:COUNT")); return; }
  uint16_t actualCrc = crc16Ccitt(payload);
  if (actualCrc != expectedCrc) {
    Serial.print(F("NAK:CRC:")); Serial.print(sequence);
    Serial.print(F(":EXPECTED:")); Serial.print(expectedCrc, HEX);
    Serial.print(F(":ACTUAL:")); Serial.print(actualCrc, HEX);
    Serial.print(F(":LEN:")); Serial.println(strlen(payload));
    return;
  }

  byte slotIndex = sequence % SLOT_COUNT;
  if (sequence < currentActive) {
    // Jangan menyentuh slot: indeks modulo yang sama mungkin sudah berisi
    // blok sequence+32. ACK ulang hanya mengonfirmasi bahwa blok lama selesai.
    Serial.print(F("BLOCK_ACK:")); Serial.print(sequence);
    Serial.println(F(":ALREADY_DONE"));
    return;
  }
  if (slotState[slotIndex] == SLOT_READY && slots[slotIndex].sequence == sequence) {
    printBlockAck(slotIndex, false);
    return;
  }
  if (slotState[slotIndex] != SLOT_FREE) {
    Serial.print(F("NAK:SLOT:")); Serial.println(sequence); return;
  }

  slotState[slotIndex] = SLOT_LOADING;
  slots[slotIndex].sequence = sequence;
  slots[slotIndex].count = count;
  char *sliceSave = NULL;
  char *slice = strtok_r(payload, ";", &sliceSave);
  byte parsed = 0;
  while (slice && parsed < count) {
    char *fieldSave = NULL;
    char *xText = strtok_r(slice, ",", &fieldSave);
    char *yText = strtok_r(NULL, ",", &fieldSave);
    char *zText = strtok_r(NULL, ",", &fieldSave);
    char *extra = strtok_r(NULL, ",", &fieldSave);
    if (!xText || !yText || !zText || extra) break;
    long dx = atol(xText), dy = atol(yText), zTarget = atol(zText);
    if (abs(dx) > ticksPerSlice || abs(dy) > ticksPerSlice) {
      slotState[slotIndex] = SLOT_FREE;
      Serial.print(F("NAK:RATE:")); Serial.println(sequence); return;
    }
    slots[slotIndex].dx[parsed] = (int16_t)dx;
    slots[slotIndex].dy[parsed] = (int16_t)dy;
    if (zTarget < -32768L || zTarget > 32767L) {
      slotState[slotIndex] = SLOT_FREE;
      Serial.print(F("NAK:Z_TARGET:")); Serial.println(sequence); return;
    }
    slots[slotIndex].zTarget[parsed] = (int16_t)zTarget;
    parsed++;
    slice = strtok_r(NULL, ";", &sliceSave);
  }
  if (parsed != count) {
    slotState[slotIndex] = SLOT_FREE;
    Serial.print(F("NAK:PAYLOAD:")); Serial.println(sequence); return;
  }

  bool resume = false;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    slotState[slotIndex] = SLOT_READY;
    if (bufferWaiting && sequence == activeSequence) resume = true;
  }
  printBlockAck(slotIndex, false);
  if (resume) {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      bufferWaiting = false;
      streamRunning = true;
      TIMSK1 |= _BV(OCIE1A);
    }
    Serial.print(F("BUFFER_RESUMED:")); Serial.println(sequence);
  }
}

void parseServo(char *line) {
  char *sequenceText = line + 6;
  char *separator1 = strchr(sequenceText, ':');
  if (!separator1) { Serial.println(F("NAK:SERVO:0:FMT")); return; }
  *separator1 = '\0';
  char *crcText = separator1 + 1;
  char *separator2 = strchr(crcText, ':');
  if (!separator2) { Serial.println(F("NAK:SERVO:0:FMT")); return; }
  *separator2 = '\0';
  char *payload = separator2 + 1;
  unsigned int sequence = (unsigned int)strtoul(sequenceText, NULL, 10);
  uint16_t receivedCrc = (uint16_t)strtoul(crcText, NULL, 16);
  if (crc16Ccitt(payload) != receivedCrc) {
    Serial.print(F("NAK:SERVO:")); Serial.print(sequence); Serial.println(F(":CRC")); return;
  }

  char *action = payload;
  bool grip;
  if (!strcmp(action, "GRIP")) grip = true;
  else if (!strcmp(action, "RELEASE")) grip = false;
  else {
    Serial.print(F("NAK:SERVO:")); Serial.print(sequence); Serial.println(F(":ACTION")); return;
  }
  byte angleValue = grip ? SERVO_GRIP_ANGLE : SERVO_RELEASE_ANGLE;

  if (lastServoValid && sequence == lastServoSequence) {
    if (grip != lastServoGrip || angleValue != lastServoAngle) {
      Serial.print(F("NAK:SERVO:")); Serial.print(sequence); Serial.println(F(":SEQUENCE_CONFLICT"));
      return;
    }
    printServoActionAck();
    if (lastServoFinished) printServoFinished();
    return;
  }
  if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled) {
    Serial.println(F("ERROR:SERVO_WHILE_RUNNING"));
    return;
  }
  startServoAction(sequence, grip, angleValue);
}

void processCommand(char *line) {
  if (!strcmp(line, "HELLO")) printReady();
  else if (!strcmp(line, "PING")) Serial.println(F("PONG"));
  else if (!strncmp(line, "STREAM_BEGIN:", 13)) {
    if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled || servoActive) {
      Serial.println(F("ERROR:ALREADY_RUNNING")); return;
    }
    if (!zEncoderInitialized || !zEncoderHealthy) {
      Serial.println(F("ERROR:Z_ENCODER")); return;
    }
    if (!zHomed) {
      Serial.println(F("ERROR:NOT_HOMED")); return;
    }
    // Parser manual lebih tahan terhadap sisa separator/karakter serial
    // daripada strtok(). Format resmi: STREAM_BEGIN:<jumlah blok>:<slice us>.
    char *totalText = line + 13;
    char *separator = strchr(totalText, ':');
    if (!separator) { Serial.println(F("ERROR:BEGIN_FORMAT")); return; }
    *separator = '\0';
    char *sliceUsText = separator + 1;
    char *trailingSeparator = strchr(sliceUsText, ':');
    if (trailingSeparator) *trailingSeparator = '\0';
    if (!*totalText || !*sliceUsText) { Serial.println(F("ERROR:BEGIN_FORMAT")); return; }
    unsigned long requestedBlocks = strtoul(totalText, NULL, 10);
    unsigned long sliceUs = strtoul(sliceUsText, NULL, 10);
    if (!requestedBlocks || sliceUs < ISR_TICK_US * 4UL || sliceUs % ISR_TICK_US) {
      Serial.println(F("ERROR:BEGIN_TIMING")); return;
    }
    resetStream();
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      totalBlocks = requestedBlocks;
      ticksPerSlice = (unsigned int)(sliceUs / ISR_TICK_US);
      tickInSlice = ticksPerSlice;
      streamConfigured = true;
    }
    Serial.print(F("STREAM_ACCEPTED:")); Serial.print(totalBlocks);
    Serial.print(F(":")); Serial.println(sliceUs);
  }
  else if (!strncmp(line, "BLOCK:", 6)) parseBlock(line);
  else if (!strcmp(line, "STREAM_RUN")) {
    if (!streamConfigured || slotState[0] != SLOT_READY) {
      Serial.println(F("ERROR:NO_INITIAL_BLOCK")); return;
    }
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      activeSequence = 0; activeSlice = 0; tickInSlice = ticksPerSlice;
      streamRunning = true; TIMSK1 |= _BV(OCIE1A);
    }
    Serial.println(F("STREAM_RUNNING"));
  }
  else if (!strcmp(line, "STOP")) {
    resetStream(); setPulseTimerInterrupt(false); stopServo();
    Serial.println(F("STREAM_ABORTED"));
  }
  else if (!strcmp(line, "STATUS")) printState(true);
  else if (!strcmp(line, "STATE?")) printState(false);
  else if (!strncmp(line, "ZCFG:", 5)) {
    if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled) {
      Serial.println(F("ERROR:ZCFG_WHILE_RUNNING")); return;
    }
    char *degText = strtok(line + 5, ":");
    char *speedText = strtok(NULL, ":");
    char *extra = strtok(NULL, ":");
    if (!degText || !speedText || extra) {
      Serial.println(F("ERROR:ZCFG_FORMAT")); return;
    }
    float newDegPerCm = atof(degText);
    float newSpeed = atof(speedText);
    if (newDegPerCm <= 0.0 || newSpeed <= 0.0) {
      Serial.println(F("ERROR:ZCFG_VALUE")); return;
    }
    zDegPerCm = newDegPerCm;
    zSpeedMmS = newSpeed;
    updateZPulseRateConfig();
    Serial.print(F("ZCFG_OK:DEG_PER_CM:")); Serial.print(zDegPerCm, 4);
    Serial.print(F(":SPEED_MM_S:")); Serial.print(zSpeedMmS, 4);
    Serial.print(F(":INTERVAL_TICKS:")); Serial.println(zConfiguredPulseIntervalTicks);
  }
  else if (!strncmp(line, "SET_POSITION:", 13)) {
    if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled) {
      Serial.println(F("ERROR:SET_POSITION_WHILE_RUNNING")); return;
    }
    char *xText = strtok(line + 13, ":");
    char *yText = strtok(NULL, ":");
    char *extra = strtok(NULL, ":");
    if (!xText || !yText || extra) {
      Serial.println(F("ERROR:SET_POSITION_FORMAT")); return;
    }
    long restoredX = atol(xText), restoredY = atol(yText);
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { positionX = restoredX; positionY = restoredY; }
    Serial.print(F("POSITION_SET:X:")); Serial.print(restoredX);
    Serial.print(F(":Y:")); Serial.println(restoredY);
  }
  else if (!strcmp(line, "HOME")) executeHome();
  else if (!strncmp(line, "SERVO:", 6)) parseServo(line);
  // Alias sederhana untuk pengujian langsung melalui Serial Monitor.
  else if (!strcmp(line, "GRIP") || !strcmp(line, "RELEASE")) {
    if (streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled) {
      Serial.println(F("ERROR:SERVO_WHILE_RUNNING"));
    } else {
      bool grip = !strcmp(line, "GRIP");
      unsigned int sequence = lastServoValid ? (unsigned int)(lastServoSequence + 1U) : 0U;
      startServoAction(
        sequence,
        grip,
        grip ? SERVO_GRIP_ANGLE : SERVO_RELEASE_ANGLE
      );
    }
  }
  else Serial.println(F("ERROR:UNKNOWN_COMMAND"));
}

void readSerial() {
  while (Serial.available()) {
    char value = (char)Serial.read();
    if (value == '\r') continue;
    if (value == '\n') {
      if (!rxOverflow && rxLength) {
        rxBuffer[rxLength] = '\0';
        processCommand(rxBuffer);
      } else if (rxOverflow) Serial.println(F("ERROR:RX_OVERFLOW"));
      rxLength = 0; rxOverflow = false;
    } else if (!rxOverflow) {
      if (rxLength < RX_BUFFER_SIZE - 1) rxBuffer[rxLength++] = value;
      else rxOverflow = true;
    }
  }
}

void serviceEvents() {
  unsigned long completedUntil = 0;
  unsigned long waitingSequence = 0;
  bool done = false, waiting = false;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    completedUntil = activeSequence;
    if (streamDonePending) { done = true; streamDonePending = false; }
    if (bufferWaitPending) {
      waiting = true;
      waitingSequence = activeSequence;
      bufferWaitPending = false;
    }
  }
  while (reportedDoneSequence < completedUntil) {
    Serial.print(F("BLOCK_DONE:")); Serial.println(reportedDoneSequence);
    reportedDoneSequence++;
  }
  if (done) Serial.println(F("STREAM_DONE"));
  if (waiting) { Serial.print(F("BUFFER_WAIT:")); Serial.println(waitingSequence); }
}

void serviceButtons() {
  bool homeNow = digitalRead(PIN_BUTTON_HOME);
  bool runNow = digitalRead(PIN_BUTTON_RUN);
  unsigned long now = millis();
  if (homeNow != lastHomeButton && now - homeChangedMs > 60) {
    lastHomeButton = homeNow; homeChangedMs = now;
    if (homeNow == LOW) { Serial.println(F("BUTTON:HOME")); executeHome(); }
  }
  if (runNow != lastRunButton && now - runChangedMs > 60) {
    lastRunButton = runNow; runChangedMs = now;
    if (runNow == LOW) {
      Serial.println(F("BUTTON:RUN"));
      if (streamConfigured && slotState[0] == SLOT_READY && !streamRunning && !bufferWaiting && !xyStreamFinished && !zPulseEnabled) {
        ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
          activeSequence = 0; activeSlice = 0; tickInSlice = ticksPerSlice;
          streamRunning = true; TIMSK1 |= _BV(OCIE1A);
        }
        Serial.println(F("STREAM_RUNNING"));
      }
    }
  }
}

void setup() {
  pinMode(PIN_STEP_X, OUTPUT); pinMode(PIN_DIR_X, OUTPUT);
  pinMode(PIN_STEP_Y, OUTPUT); pinMode(PIN_DIR_Y, OUTPUT);
  pinMode(PIN_STEP_Z, OUTPUT); pinMode(PIN_DIR_Z, OUTPUT);
  pinMode(PIN_SERVO_GRIP, OUTPUT);
  pinMode(PIN_AS5600_POWER, OUTPUT);
  pinMode(PIN_LIMIT_X, INPUT_PULLUP); pinMode(PIN_LIMIT_Y, INPUT_PULLUP); pinMode(PIN_LIMIT_Z, INPUT_PULLUP);
  pinMode(PIN_BUTTON_HOME, INPUT_PULLUP); pinMode(PIN_BUTTON_RUN, INPUT_PULLUP);
  digitalWrite(PIN_STEP_X, LOW); digitalWrite(PIN_STEP_Y, LOW); digitalWrite(PIN_STEP_Z, LOW);
  digitalWrite(PIN_SERVO_GRIP, LOW);
  digitalWrite(PIN_AS5600_POWER, HIGH);
  delay(20);
  Serial.begin(250000);
  Wire.begin();
  Wire.setWireTimeout(3000UL, true);
  Wire.setClock(400000UL);
  updateZEncoder();
  updateZPulseRateConfig();
  configureTimer1();
  printReady();
}

void loop() {
  readSerial();
  serviceEvents();
  serviceButtons();
  serviceServo();
  serviceZClosedLoop();
  if (zFaultPending) {
    zFaultPending = false;
    bool motionActive = streamRunning || bufferWaiting || xyStreamFinished || zPulseEnabled;
    if (motionActive) {
      resetStream();
      setPulseTimerInterrupt(false);
      zProgressTracking = false;
    }
    Serial.println(F("ERROR:Z_ENCODER_READ"));
  }
}
