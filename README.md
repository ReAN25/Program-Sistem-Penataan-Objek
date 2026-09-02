# Program Sistem Penataan Objek

Program ini memakai GUI pembuat trajektori dari Segment Planner, tetapi cara
eksekusinya berbeda. Python membentuk timeline blok berisi perubahan pulsa X/Y
dan target sudut Z. Arduino menjalankan DDA X/Y, closed-loop Z dengan AS5600,
serta servo gripper.


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
