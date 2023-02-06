import sys

out = open(sys.argv[2], "w")
out.write("graph {\n")
for line in open(sys.argv[1], "r").readlines()[1:]:
    u, v = map(int, line.strip().split())
    out.write(str(u) + "--" + str(v) + ";\n")

out.write("}\n")

out.close()

