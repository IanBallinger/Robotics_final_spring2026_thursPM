# Robotics_final_spring2026_thursPM

Note: ur_rtde only works with python 3.12 on windows. 

## Getting Started:
from your terminal:

```bash
git clone https://github.com/IanBallinger/Robotics_final_spring2026_thursPM.git
cd Robotics_final_spring2026_thursPM
git submodule update --init --recursive
code .
```

Then (windows):
winget ships with win11, but you can also install it from [it's github releases page](https://github.com/microsoft/winget-cli/releases)
```bash
winget install Python.Python.3.12
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

(linux/macos):
You should have python 3.12 installed.
```bash
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```