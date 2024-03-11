from subprocess import Popen, PIPE

test = Popen(['./rk4 %s %s %s %s %s %s' %(str(1), str(2), str(3), str(4), str(5), str(0.2))], shell=True, stdout=PIPE, stdin=PIPE).communicate()[0]

print(test)

test = test.decode('utf-8')

print(test)

test = test.split(',')

print(test)

output = [float(n) for n in test]

print(output)