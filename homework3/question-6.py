def ji(x, y, z):
    xx = (13 - y + z) / 4
    yy = (x - z + 8) / 5
    zz = (2 * x - y + 2) / 6
    return xx, yy, zz


def gsi(x, y, z):
    xx = (13 - y + z) / 4
    yy = (xx - z + 8) / 5
    zz = (2 * xx - yy + 2) / 6
    return xx, yy, zz


print("Jacobi iteration:")


x = 0
y = 0
z = 0
print(x, y, z)
for i in range(3):
    x, y, z = ji(x, y, z)
    print(x, y, z)

print("Gauss-Seidel iteration:")
x = 0
y = 0
z = 0
print(x, y, z)
for i in range(3):
    x, y, z = gsi(x, y, z)
    print(x, y, z)
