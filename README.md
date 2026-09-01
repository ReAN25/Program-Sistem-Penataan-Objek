# Program Pulse Block Streaming XYZ-angle + Servo

Program ini memakai GUI pembuat trajektori dari Segment Planner, tetapi cara
eksekusinya berbeda. Python membentuk timeline blok berisi perubahan pulsa X/Y
dan target sudut Z. Arduino menjalankan DDA X/Y, closed-loop Z dengan AS5600,
serta servo gripper.

Entry point produksi adalah `run_gui.bat` yang memuat `full_gui/`. Jalur produksi
hanya menggunakan `pulse_planner.py`, `stream_transport.py`, `pulse_serial.py`,
dan `pulse_execution.py` bersama mixin GUI. `python_app/` dipertahankan khusus
sebagai GUI uji/offline dan `run_stream_test.bat`; modul protokol lama
`execution.py` dan `serial_comm.py` sudah dihapus karena tidak pernah diimpor.

Modul FHO dan penyusunan rute produksi diselaraskan dengan Segment Planner:
10 control point per segmen, 20 kandidat (6 Fire Hawk dan 14 Prey), 300 epoch,
baseline bebas collision, objective panjang/kehalusan/clearance, serta
preparasi titik 0,3 mm dan urutan Grip–Release yang sama. Setiap eksekusi juga
menghasilkan CSV `run_cycle_*` di `run_logs`, termasuk ketika dihentikan di
tengah.

## Konfigurasi gerak

- Kecepatan X, Y, dan Z diinput terpisah dalam mm/s.
- Durasi segmen mengikuti sumbu yang memerlukan waktu paling lama agar XYZ
  mencapai target pada timeline yang sama.
- Kalibrasi X/Y memakai pulsa/cm. Kalibrasi Z default 15°/cm.
- AS5600 berada di poros output gearbox. Python mengirim target sudut output
  dalam 0,1°; Arduino membaca dan melacak sudut output secara multi-turn tanpa
  mengalikannya dengan 20. Reduksi 20:1 hanya dipakai untuk menghitung pulsa
  motor yang diperlukan untuk mencapai sudut output tersebut.
- Timeline gerak memakai slice 10 ms dan setiap blok memuat dua slice (20 ms);
  pembacaan feedback AS5600 dilakukan setiap 20 ms.
- Selama arah error Z belum berubah, generator pulsa tidak di-reset pada setiap
  sampel encoder; counter timer berjalan kontinu untuk mengurangi jeda dan
  efek tersendat.
- Firmware mempunyai 32 slot buffer.
- ISR 20 µs menjalankan DDA X/Y dan generator pulsa Z; STEP Z dipertahankan
  HIGH selama satu tick (20 µs) dengan LOW minimal satu tick.
- Homing X, Y, dan Z memakai HIGH/LOW masing-masing 30 µs.
- Arah mekanis sumbu Y dibalik melalui satu konfigurasi firmware yang sama
  untuk gerak streaming dan homing; tanda koordinat Y di Python tetap tidak
  berubah.
- Sebelum RUN, nilai Offset Gripper Z Axis dibaca ulang dari GUI. Setiap
  checkpoint GRIP/RELEASE menjalankan gerak `ACTION_GOTO` ke
  `Z_trajektori + offset`, menunggu konfirmasi servo, kemudian kembali ke
  titik trajektori.
- Untuk rute direct, smoothing dibagi pada waypoint GRIP/RELEASE sehingga
  checkpoint internal tidak bergeser oleh moving average.
- Planner memakai timeline linear dengan kecepatan konstan pada setiap slice;
  fungsi percepatan dan perlambatan telah dihilangkan baik pada **CYCLE OFF**
  maupun **CYCLE ON**. Perubahan arah mengikuti geometri polyline tanpa profil
  ramp waktu.
- Tombol **Generate Trajectory** (tanpa optimasi) menghitung jarak setiap
  segmen secara terpisah dengan jarak Euclidean antar titik berurutan,
  termasuk segmen dari titik homing `(0,0,0)` ke target pertama. Hasil setiap
  segmen, total lintasan, serta akumulasi perpindahan absolut X, Y, dan Z
  ditampilkan pada log GUI dalam mm.
- Tombol **Save Trajektory** menyimpan plot utama, CSV bundle, dan gambar
  terpisah untuk setiap pasangan jalur Grip--Release. Gambar berada pada folder
  bertanda waktu dengan nama berawalan `S1A`, `S1B`, hingga pasangan terakhir
  (misalnya `S1A_ISO1.png` atau `S1A_XZ_1.png`). Setiap jalur memiliki delapan
  PNG terpisah: empat sudut isometrik dan empat tampak samping dari arah
  berlawanan pada bidang X-Z dan Y-Z.
- Gerakan hanya berhenti pada checkpoint Grip/Release agar aksi servo dapat
  dijalankan dan dikonfirmasi.

## Servo

Sudut servo tidak menjadi input GUI. Nilainya tetap seperti Segment Planner:
Grip 30° dan Release 95°. Paket servo berisi nomor urut, CRC16, dan jenis aksi.
Retry dengan nomor yang sama bersifat idempoten:
Arduino hanya mengirim ulang ACK dan tidak menggerakkan servo untuk kedua kali.
Konfirmasi `GOK`/`ROK` dikirim setelah waktu settle servo selesai.

## Pemulihan error

- CRC/format blok rusak: blok yang sama dikirim ulang.
- `ERROR:BEGIN_FORMAT` pada pembukaan stream dianggap frame serial sementara;
  GUI mengirim ulang `STREAM_BEGIN` yang sama sampai Arduino menerima. Hanya
  `ERROR:BEGIN_TIMING` yang langsung ditolak karena parameter waktunya salah.
- ACK blok hilang: GUI meminta `STATE?`; jika Arduino menyatakan blok sudah
  aktif/selesai, GUI melanjutkan tanpa mengirim gerakan ganda.
- `BUFFER_WAIT`: X/Y berhenti pada batas blok dan melanjutkan sequence yang sama
  setelah blok valid diterima. Waktu tunggu ini tidak dihitung sebagai
  `Z_STALL` karena Timer1 memang dihentikan sementara.
- Akhir blok/checkpoint menunggu galat Z berada dalam toleransi 0,5° sebelum
  `STREAM_DONE` atau aksi servo dijalankan.
- Kehilangan pembacaan AS5600 menghentikan pulsa Z dan menghasilkan
  `ERROR:Z_ENCODER_READ`; watchdog juga membatalkan stream jika Z tidak maju
  selama 3 detik (`ERROR:Z_STALL`).
- Jika feedback sudut bergerak berlawanan dengan arah error target, firmware
  menghentikan stream dalam 250 ms dan mengirim `ERROR:Z_WRONG_DIRECTION`.
- Jika ACK, STATE, PING, atau jawaban homing hilang, perintah yang sama dikirim
  ulang. Sebelum pengulangan blok, GUI memakai sequence/CRC dan STATE agar
  gerakan tidak dijalankan dua kali.
- Sequence blok memakai bilangan 32-bit. Objek blok Python yang sudah menerima
  `BLOCK_DONE` dilepas dari cache; bila Arduino mengirim `BUFFER_WAIT`, blok
  tersebut dibangun ulang dari data slice dan dikirim dengan sequence yang sama.
- Uji software-in-the-loop trajektori panjang dapat dijalankan dengan
  `python tools/stress_test_long_stream.py`; uji ini menyuntikkan CRC/COUNT,
  timeout ACK, RX overflow, dan BUFFER_WAIT tanpa Arduino.
- ACK servo hilang: GUI memeriksa `GRIP` dan `SERVO` pada STATE. Servo tidak
  diulang jika posisi target sudah benar atau servo masih bergerak.
- STOP hanya dikirim karena perintah operator atau sebelum gerak manual ketika
  STATE memastikan RUN lama memang masih aktif.

## Firmware dan pin

Upload `arduino_pulse_stream/arduino_pulse_stream.ino` untuk Arduino Uno.

- STEP/DIR X: 8/9
- STEP/DIR Y: 6/7
- STEP/DIR Z: 4/5
- Servo gripper: 2
- Limit X/Y/Z: 11/12/13
- AS5600: I²C A4/A5
- Tombol HOME/RUN: A1/A2
- Baud rate: 250000

Identitas firmware:

`PULSE_BLOCK_READY:32:SEQ_BITS:32:BLOCK_SLICES:2:SLOTS:32:TICK_US:20:AXES:XYZ_ANGLE:Y_DIR_INVERTED:1:Z_HIGH_US:20:Z_AS5600:1:Z_ENCODER_OUTPUT:1:Z_SAMPLE_MS:20:REDUCTION:20:SERVO_FIXED:1:SERVO_GRIP_MS:1000:STATE_RECOVERY:1`

## Menjalankan

```powershell
cd "G:\My Drive\SKRIPSI\Program\PROGRAM PULSE BLOCK STREAMING"
python -m pip install -r requirements.txt
.\run_gui.bat
```

Pengujian:

```powershell
python -m unittest discover -s tests -v
arduino-cli compile --fqbn arduino:avr:uno .\arduino_pulse_stream
```
