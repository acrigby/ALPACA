from subprocess import Popen, PIPE

test = Popen(["./rk4 1 2 3 4 5 6"], shell=True, stdout=PIPE, stdin=PIPE).communicate()[0]

print(test)

