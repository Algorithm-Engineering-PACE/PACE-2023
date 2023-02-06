import os

for i in range(2, 70, 2):
    inp = "exact_%03d.dot" % i
    out = "exact_%03d.pdf" % i
    os.system("neato -Tpdf -Goverlap=scale " + inp + " > " + out)
