from pathlib import Path

db = Path(__file__).parent.parent

for file in db.glob("*.dat"):
    with open(file, "r") as f:
        s = f.read()

    s_fixed = s.encode("ascii", "ignore").decode()

    if s != s_fixed:
        with open(file, "w+") as f:
            f.write(s_fixed)