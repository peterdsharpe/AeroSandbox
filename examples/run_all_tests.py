# Beware - running this file will take ages!
# Runs all examples so that you can determine if any examples are broken.
import os


for filename in os.listdir():
    if ".py" in filename and filename != "run_all_tests.py":
        print("Running %s..." % filename)
        os.system("python %s" % filename)