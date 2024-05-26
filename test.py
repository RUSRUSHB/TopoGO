import sympy as sp

c1, s1, c2, s2, c3, s3, c23, s23, l1, l2, l3 = sp.symbols('c1 s1 c2 s2 c3 s3 c23 s23 l1 l2 l3')

# 生成矩阵

matR = sp.Matrix([[c1*c23, s1*c23, s23], [-c1*s23, -s1*s23, c23], [s1, -c1, 0]])



matJv = sp.Matrix([[-s1*(l1+l2*c2+l3*c23), -l2*c1*s2, -l3*c1*s23], [c1*(l1+l2*c2+l3*c23), l2*s1*c2, -l3*s1*s23], [0, l2*c2, l3*c23]])
matJo = sp.Matrix([[0, 0, s1], [0, -1, -c1], [1, 0, 0]])
# 矩阵乘法


matJv = matR * matJv
matJo = matR * matJo
# 逐行打印矩阵
for i in range(3):
    print(matJv[i, :])

for i in range(3):
    print(matJo[i, :])