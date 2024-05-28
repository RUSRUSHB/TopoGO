import sympy as sp

def process_polynomial(poly):
    t = sp.Symbol('t')
    
    # 获取多项式的常数项
    const_term = poly.as_coefficients_dict().get(t**0, 0)
    
    if const_term != 0:
        # 如果常数项非零
        if const_term > 0:
            return poly
        else:
            return -poly
    else:
        # 如果常数项为零
        # 获取最低次幂项和其系数
        terms = poly.as_ordered_terms()
        lowest_term = terms[-1]
        lowest_coeff, lowest_exp = lowest_term.as_coeff_exponent(t)
        
        if lowest_coeff < 0:
            poly = -poly

        # 移除最低次幂的幂但保留其系数
        lowest_monom = t**lowest_exp
        poly = sp.div(poly, lowest_monom)[0]

        
        return poly

# 示例多项式
t = sp.Symbol('t')
poly1 = -2*t**3 + 3*t**2 - 2*t
poly2 = t**4
poly3 = t + 1
poly4 = t - 1

# 处理并打印结果
print(process_polynomial(poly1))  # 输出: 2*t**2 - 3*t + 2
print(process_polynomial(poly2))  # 输出: 1
print(process_polynomial(poly3))  # 输出: t + 1
print(process_polynomial(poly4))  # 输出: -t + 1
